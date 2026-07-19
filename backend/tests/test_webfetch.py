"""Web-source ingestion tests (offline: httpx MockTransport, no network).

Locks in the behaviours that matter for evidence fetching:
  - a realistic browser header profile is sent (many public sites WAF-block requests
    that lack the Accept / Sec-Fetch set — a bare User-Agent gets 403);
  - HTTP errors raise (so the API layer records them as a skipped-URL note);
  - HTML is reduced to readable text.
"""

import httpx

from chain_of_claims.ingest.webfetch import fetch_url_text, html_to_text


def _client_patch(monkeypatch, handler):
    """Route httpx.Client(...) through a MockTransport running `handler`."""
    real_client = httpx.Client

    def factory(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "Client", factory)


def test_sends_browser_headers(monkeypatch):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(request.headers)
        return httpx.Response(200, headers={"content-type": "text/html"},
                              text="<html><body><p>Net value grew.</p></body></html>")

    _client_patch(monkeypatch, handler)
    out = fetch_url_text("https://example.com/report")
    # the header profile that clears bot walls must be present
    assert "mozilla/5.0" in seen.get("user-agent", "").lower()
    assert "text/html" in seen.get("accept", "")
    assert seen.get("accept-language")
    assert seen.get("sec-fetch-mode") == "navigate"
    assert "Net value grew." in out


def test_http_error_raises(monkeypatch):
    # A 403 (the Temasek WAF case, if headers were insufficient) must raise so the API
    # records a skipped-URL note rather than silently ingesting an error page.
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, text="Forbidden")

    _client_patch(monkeypatch, handler)
    try:
        fetch_url_text("https://blocked.example/report")
        assert False, "expected an HTTPStatusError"
    except httpx.HTTPStatusError:
        pass


def test_html_to_text_reduces_markup():
    raw = (
        "<html><head><style>.x{}</style></head><body>"
        "<h1>Title</h1><p>Revenue rose 10%.</p>"
        "<script>evil()</script><p>Costs held flat.</p></body></html>"
    )
    txt = html_to_text(raw)
    assert "Title" in txt
    assert "Revenue rose 10%." in txt
    assert "Costs held flat." in txt
    assert "evil()" not in txt  # script content dropped
    assert "<" not in txt       # tags stripped
