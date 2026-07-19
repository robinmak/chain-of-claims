"""Fetch web pages and reduce them to plain text for use as evidence.

Dependency-light: uses httpx (already present via FastAPI) for the request and a small
regex-based HTML-to-text reducer rather than a full parser. This is sufficient for the
verification use case, where we only need the page's textual content as candidate
evidence, not a faithful DOM. Fetches are best-effort: a URL that fails to load is
skipped with a note rather than failing the whole run.
"""

from __future__ import annotations

import html
import re

# Strip these elements entirely (content included) before extracting text.
_DROP_BLOCKS = re.compile(
    r"<(script|style|noscript|template|svg|head)\b[^>]*>.*?</\1>",
    re.IGNORECASE | re.DOTALL,
)
_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"[ \t\r\f\v]+")
_BLANKS = re.compile(r"\n\s*\n\s*\n+")
# Turn block-level closes into paragraph breaks so chunking sees paragraphs.
_BLOCK_END = re.compile(
    r"</(p|div|section|article|li|h[1-6]|tr|table|ul|ol|header|footer)\s*>",
    re.IGNORECASE,
)
_BR = re.compile(r"<br\s*/?>", re.IGNORECASE)


def html_to_text(raw: str) -> str:
    s = _DROP_BLOCKS.sub(" ", raw)
    s = _BR.sub("\n", s)
    s = _BLOCK_END.sub("\n\n", s)
    s = _TAG.sub("", s)
    s = html.unescape(s)
    s = _WS.sub(" ", s)
    s = "\n".join(line.strip() for line in s.splitlines())
    s = _BLANKS.sub("\n\n", s)
    return s.strip()


def fetch_url_text(url: str, *, timeout: float = 20.0, max_chars: int = 200_000) -> str:
    """Fetch `url` and return extracted plain text. Raises on network/HTTP error.

    Sends a realistic browser header profile: many public sites (news, corporate
    IR) sit behind WAFs that return 403 to requests missing the Accept / Sec-Fetch
    header set a browser sends, even when the page itself is public. A bare UA is not
    enough — the full profile is what clears the bot wall for legitimately public
    evidence pages.
    """
    import httpx

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": (
            "text/html,application/xhtml+xml,application/xml;q=0.9,"
            "image/avif,image/webp,*/*;q=0.8"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
    }
    with httpx.Client(follow_redirects=True, timeout=timeout, headers=headers) as client:
        resp = client.get(url)
        resp.raise_for_status()
        ctype = resp.headers.get("content-type", "")
        body = resp.text
    text = html_to_text(body) if "html" in ctype.lower() or "<" in body[:200] else body
    return text[:max_chars]
