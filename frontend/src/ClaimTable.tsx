import { useState, Fragment } from "react";
import { ClaimView } from "./types";
import { ClaimEvidenceGraph } from "./ClaimEvidenceGraph";

const CITATION_BADGE: Record<string, string> = {
  supported: "good",
  uncited_supported: "warn",
  out_of_context: "bad",
  uncited_unsupported: "bad",
  not_applicable: "muted",
};

const FACT_BADGE: Record<string, string> = {
  supported: "good",
  refuted: "bad",
  not_enough_evidence: "warn",
  not_checked: "muted",
};

function Badge({ value, map }: { value: string; map: Record<string, string> }) {
  return <span className={`badge ${map[value] || "muted"}`}>{value}</span>;
}

function CQBar({ rates }: { rates: number[] }) {
  // one cell per critical question; green if panel mostly passed it
  return (
    <div className="cqbar" title="Per critical-question pass rate (8 CQs)">
      {rates.map((r, i) => (
        <div
          key={i}
          className="cqcell"
          style={{ background: r >= 0.5 ? "var(--good)" : "var(--bad)", opacity: 0.4 + r * 0.6 }}
        />
      ))}
    </div>
  );
}

function ClaimDetail({ c }: { c: ClaimView }) {
  const [view, setView] = useState<"graph" | "table">("graph");
  return (
    <div>
      <div style={{ display: "flex", gap: 8, margin: "4px 0 8px" }}>
        <button
          onClick={() => setView("graph")}
          style={{ background: view === "graph" ? "var(--accent)" : "var(--panel-2)",
                   color: view === "graph" ? "#06122b" : "var(--text)", padding: "4px 12px" }}
        >
          Graph
        </button>
        <button
          onClick={() => setView("table")}
          style={{ background: view === "table" ? "var(--accent)" : "var(--panel-2)",
                   color: view === "table" ? "#06122b" : "var(--text)", padding: "4px 12px" }}
        >
          Details
        </button>
      </div>
      {view === "graph" ? <ClaimEvidenceGraph c={c} /> : <ClaimDetailTable c={c} />}
    </div>
  );
}

function ClaimDetailTable({ c }: { c: ClaimView }) {
  return (
    <div className="detail">
      {c.triplet && (
        <>
          <h4>Toulmin structure</h4>
          <div className="kv"><b>Reason:</b> {c.triplet.reason}</div>
          <div className="kv">
            <b>Warrant</b> {c.triplet.warrant_generated ? "(generated)" : ""}
            {c.triplet.is_causal ? " · causal" : ""}: {c.triplet.warrant}
          </div>
        </>
      )}
      {c.warrant_audit && (
        <>
          <h4 style={{ marginTop: 10 }}>Warrant audit (critical questions)</h4>
          <div className="kv">
            CQ score {Math.round(c.warrant_audit.cq_score * 100)}% · panel agreement{" "}
            {Math.round(c.warrant_audit.cq_agreement * 100)}%
            {c.warrant_audit.structural_pass !== null &&
              ` · causal structure ${c.warrant_audit.structural_pass ? "pass" : "fail"}`}
          </div>
          <CQBar rates={c.warrant_audit.per_question_pass_rate} />
          {c.warrant_audit.causal_pairs && c.warrant_audit.causal_pairs.length > 0 && (
            <div className="kv" style={{ marginTop: 8 }}>
              <b>Causal links extracted:</b>
              <ul style={{ margin: "4px 0" }}>
                {c.warrant_audit.causal_pairs.map((p, i) => {
                  const v = c.warrant_audit?.causal_attribution?.[i];
                  return (
                    <li key={i}>
                      {p.cause} → {p.effect}
                      {v && ` · evidence: ${v.attribution.replace(/_/g, " ")}`}
                    </li>
                  );
                })}
              </ul>
              {c.warrant_audit.causal_attribution_score !== null && (
                <span>
                  Attribution score{" "}
                  {Math.round(c.warrant_audit.causal_attribution_score * 100)}%
                  {" "}(diagnostic — not scored)
                </span>
              )}
            </div>
          )}
        </>
      )}
      <h4 style={{ marginTop: 10 }}>
        Evidence ({c.evidence.length}) {c.verdict?.method_used ? `· method: ${c.verdict.method_used}` : ""}
      </h4>
      {c.evidence.length === 0 && <div className="kv">No grounding evidence found in sources.</div>}
      {c.evidence.map((e, i) => (
        <div className="evidence-item" key={i}>
          <b>{e.source}</b> [{e.locator}] {e.text}
        </div>
      ))}
      {c.verdict?.detail && (
        <div className="kv" style={{ marginTop: 8 }}>
          <b>Verifier note:</b> {c.verdict.detail}
        </div>
      )}
    </div>
  );
}

export function ClaimTable({ claims }: { claims: ClaimView[] }) {
  const [open, setOpen] = useState<number | null>(null);
  return (
    <div className="card">
      <h4 style={{ marginTop: 0 }}>Claims ({claims.length})</h4>
      <table>
        <thead>
          <tr>
            <th>Claim</th>
            <th>Type</th>
            <th>Fact-check</th>
            <th>Citation</th>
            <th>Evidence</th>
          </tr>
        </thead>
        <tbody>
          {claims.map((c) => (
            <Fragment key={c.id}>
              <tr className="clickable" onClick={() => setOpen(open === c.id ? null : c.id)}>
                <td>{c.text}</td>
                <td className="type-tag">{c.type}</td>
                <td>
                  {c.verdict ? <Badge value={c.verdict.factcheck_result} map={FACT_BADGE} /> : "-"}
                </td>
                <td>
                  {c.verdict ? <Badge value={c.verdict.citation_status} map={CITATION_BADGE} /> : "-"}
                </td>
                <td>{c.evidence.length}</td>
              </tr>
              {open === c.id && (
                <tr>
                  <td colSpan={5}>
                    <ClaimDetail c={c} />
                  </td>
                </tr>
              )}
            </Fragment>
          ))}
        </tbody>
      </table>
    </div>
  );
}
