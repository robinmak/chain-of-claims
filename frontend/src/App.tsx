import { useRef, useState } from "react";
import { RunResult } from "./types";
import { ScoresView } from "./Scores";
import { ClaimTable } from "./ClaimTable";

export function App() {
  const reportRef = useRef<HTMLInputElement>(null);
  const sourcesRef = useRef<HTMLInputElement>(null);
  const goldRef = useRef<HTMLInputElement>(null);
  const [docMode, setDocMode] = useState<"file" | "text">("file");
  const [srcMode, setSrcMode] = useState<"file" | "text" | "url">("file");
  const [docText, setDocText] = useState("");
  const [srcText, setSrcText] = useState("");
  const [srcUrls, setSrcUrls] = useState("");
  const [running, setRunning] = useState(false);
  const [log, setLog] = useState<string[]>([]);
  const [result, setResult] = useState<RunResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function start() {
    setError(null);
    setResult(null);
    setLog([]);

    const fd = new FormData();

    // Document: uploaded file OR pasted text.
    if (docMode === "file") {
      const report = reportRef.current?.files?.[0];
      if (!report) {
        setError("Select a document file, or switch to Paste text.");
        return;
      }
      fd.append("report", report);
    } else {
      if (!docText.trim()) {
        setError("Enter the statement or text to analyze.");
        return;
      }
      fd.append("report_text", docText);
    }

    // Sources (optional): uploaded files, pasted text, OR web URLs.
    if (srcMode === "file") {
      const sources = sourcesRef.current?.files;
      if (sources) for (const s of Array.from(sources)) fd.append("sources", s);
    } else if (srcMode === "text") {
      if (srcText.trim()) fd.append("source_text", srcText);
    } else if (srcMode === "url") {
      if (srcUrls.trim()) fd.append("source_urls", srcUrls);
    }

    const gold = goldRef.current?.files?.[0];
    if (gold) fd.append("gold", gold);

    setRunning(true);
    try {
      const res = await fetch("/runs", { method: "POST", body: fd });
      const { run_id } = await res.json();
      const es = new EventSource(`/runs/${run_id}/stream`);
      es.addEventListener("progress", (e: MessageEvent) => {
        setLog((l) => [...l, e.data]);
      });
      es.addEventListener("end", async () => {
        es.close();
        const r = await fetch(`/runs/${run_id}`);
        const data: RunResult = await r.json();
        setResult(data);
        if (data.status === "failed") setError(data.error || "Run failed");
        setRunning(false);
      });
      es.onerror = () => {
        es.close();
        setRunning(false);
      };
    } catch (e) {
      setError(String(e));
      setRunning(false);
    }
  }

  return (
    <div className="wrap">
      <h1>Chain-of-Claims</h1>
      <div className="sub">
        Claim–evidence verification for LLM-generated financial-services documents —
        research notes, filings, investment memos, shareholder letters, and disclosures.
        Upload or paste a document — a full report, or a single statement, paragraph, or
        section — and add the source materials it should be checked against.
      </div>

      <div className="card">
        <div className="upload-row">
          <div className="field">
            <label>Document</label>
            <ModeToggle
              name="doc-mode"
              mode={docMode}
              options={[
                { value: "file", label: "Upload" },
                { value: "text", label: "Paste text" },
              ]}
              onChange={setDocMode}
            />
            {docMode === "file" ? (
              <input ref={reportRef} type="file" accept=".pdf,.txt,.md,.markdown" />
            ) : (
              <textarea
                className="paste"
                placeholder="Paste a statement, paragraph, or section to verify…"
                value={docText}
                onChange={(e) => setDocText(e.target.value)}
              />
            )}
          </div>
          <div className="field">
            <label>Source materials (optional)</label>
            <ModeToggle
              name="src-mode"
              mode={srcMode}
              options={[
                { value: "file", label: "Upload" },
                { value: "text", label: "Paste text" },
                { value: "url", label: "Website" },
              ]}
              onChange={setSrcMode}
            />
            {srcMode === "file" && (
              <input ref={sourcesRef} type="file" accept=".pdf,.txt,.md,.markdown" multiple />
            )}
            {srcMode === "text" && (
              <textarea
                className="paste"
                placeholder="Paste supporting evidence (figures, filings, facts)…"
                value={srcText}
                onChange={(e) => setSrcText(e.target.value)}
              />
            )}
            {srcMode === "url" && (
              <textarea
                className="paste"
                placeholder="Enter web URLs to fetch as evidence, one per line…"
                value={srcUrls}
                onChange={(e) => setSrcUrls(e.target.value)}
              />
            )}
          </div>
          <div className="field">
            <label>Gold claims (optional, .json — enables coverage)</label>
            <input ref={goldRef} type="file" accept=".json" />
          </div>
        </div>
        <div className="verify-row">
          <button onClick={start} disabled={running}>
            {running ? "Analyzing…" : "Verify claims"}
          </button>
        </div>
      </div>

      {error && (
        <div className="card" style={{ borderColor: "var(--bad)" }}>
          <b style={{ color: "var(--bad)" }}>Error:</b> {error}
        </div>
      )}

      {(running || log.length > 0) && <ProgressStepper log={log} running={running} />}

      {result?.scores && <ScoresView scores={result.scores} />}
      {result && result.claims.length > 0 && <ClaimTable claims={result.claims} />}
    </div>
  );
}

// Canonical pipeline stages in execution order, mapped to user-facing labels. The
// backend emits SSE lines as "<stage_key>: <message>"; some keys cover two numbered
// stages (s3_toulmin = Toulmin + warrant audit; s8_factcheck = fact-check + citation).
// `optional` steps (gold coverage, causal attribution) only run in some configs.
const STEPS: { key: string; label: string; optional?: boolean }[] = [
  { key: "ingest", label: "Ingest source materials" },
  { key: "s1_extract", label: "Extract atomic claims" },
  { key: "s2_prune", label: "Prune duplicates & boilerplate" },
  { key: "s1b_coverage", label: "Coverage vs. gold set", optional: true },
  { key: "s3_toulmin", label: "Argument structure & warrant audit" },
  { key: "s6_ground", label: "Ground claims in evidence" },
  { key: "s4b_causal", label: "Causal attribution", optional: true },
  { key: "s7_explainability", label: "Explainability" },
  { key: "s8_factcheck", label: "Fact-check & citations" },
  { key: "s10_hallucination", label: "Score" },
];

type StepStatus = "done" | "active" | "pending" | "skipped";

function ProgressStepper({ log, running }: { log: string[]; running: boolean }) {
  // Parse "stage: message" lines into (stage, message) and derive state.
  const parsed = log.map((line) => {
    const idx = line.indexOf(": ");
    return idx === -1
      ? { stage: line, message: "" }
      : { stage: line.slice(0, idx), message: line.slice(idx + 2) };
  });

  const seen = new Set(parsed.map((p) => p.stage));
  const doneRun = seen.has("done") || !running;
  const stepIndexByKey = new Map(STEPS.map((s, i) => [s.key, i]));
  // Latest canonical step that has been reported.
  let latestIdx = -1;
  for (const p of parsed) {
    const i = stepIndexByKey.get(p.stage);
    if (i !== undefined && i > latestIdx) latestIdx = i;
  }

  const statusFor = (i: number, key: string): StepStatus => {
    if (seen.has(key)) return i === latestIdx && !doneRun ? "active" : "done";
    if (doneRun || i < latestIdx) return "skipped"; // never ran (no gold / no sources / config)
    return "pending";
  };

  // Latest human-readable message, shown as a caption under the active step.
  const latest = parsed[parsed.length - 1];
  // webfetch notes (e.g. skipped URLs) are surfaced separately — they matter.
  const notes = parsed.filter((p) => p.stage === "webfetch").map((p) => p.message);

  return (
    <div className="card">
      <div className="stepper">
        {STEPS.map((s, i) => {
          const st = statusFor(i, s.key);
          return (
            <div key={s.key} className={`step ${st}`}>
              <span className="step-icon" aria-hidden>
                {st === "done" && "✓"}
                {st === "active" && <span className="spinner" />}
                {st === "skipped" && "–"}
                {st === "pending" && ""}
              </span>
              <span className="step-label">
                {s.label}
                {s.optional && st === "skipped" && (
                  <span className="step-skip"> · not applicable</span>
                )}
              </span>
            </div>
          );
        })}
      </div>
      {running && latest && (
        <div className="step-caption">{latest.message || "Starting…"}</div>
      )}
      {notes.length > 0 && (
        <div className="step-notes">
          {notes.map((n, i) => (
            <div key={i} className="step-note">⚠ {n}</div>
          ))}
        </div>
      )}
      {log.length > 0 && (
        <details className="step-rawlog">
          <summary>Show full log</summary>
          <div className="progress">{log.join("\n")}</div>
        </details>
      )}
    </div>
  );
}

function ModeToggle<M extends string>({
  name,
  mode,
  options,
  onChange,
}: {
  name: string;
  mode: M;
  options: { value: M; label: string }[];
  onChange: (m: M) => void;
}) {
  return (
    <div className="mode-toggle">
      {options.map((opt) => (
        <label key={opt.value}>
          <input
            type="radio"
            name={name}
            checked={mode === opt.value}
            onChange={() => onChange(opt.value)}
          />
          {opt.label}
        </label>
      ))}
    </div>
  );
}
