import { Scores } from "./types";

export function ScoresView({ scores }: { scores: Scores }) {
  const skipped = scores.verification_skipped;
  const explPct = scores.explainability !== null ? Math.round(scores.explainability * 100) : null;
  const halluPct = scores.hallucination !== null ? Math.round(scores.hallucination * 100) : null;
  return (
    <div className="card">
      {skipped && (
        <div
          className="sub"
          style={{ marginTop: 0, marginBottom: 14, color: "var(--warn)" }}
        >
          No source materials supplied — claims and argument structure were extracted, but
          grounding, fact-checking, and citation verification were skipped. Add source
          materials to score explainability and hallucination.
        </div>
      )}
      <div className="scores">
        <div className="score-tile">
          <div className="label">Explainability (grounding coverage)</div>
          <div className="value" style={{ color: explPct !== null ? "var(--good)" : "var(--muted)" }}>
            {explPct !== null ? `${explPct}%` : "N/A"}
          </div>
        </div>
        <div className="score-tile">
          <div className="label">Hallucination rate</div>
          <div
            className="value"
            style={{
              color: halluPct === null ? "var(--muted)" : halluPct > 20 ? "var(--bad)" : "var(--warn)",
            }}
          >
            {halluPct !== null ? `${halluPct}%` : "N/A"}
          </div>
        </div>
        <div className="score-tile">
          <div className="label">Claims (check-worthy / total)</div>
          <div className="value">{scores.n_checkworthy}/{scores.n_claims}</div>
        </div>
        {scores.coverage !== null && (
          <div className="score-tile">
            <div className="label">Coverage (recall vs. gold)</div>
            <div className="value" style={{ color: scores.coverage >= 0.9 ? "var(--good)" : "var(--warn)" }}>
              {Math.round(scores.coverage * 100)}%
            </div>
            <div className="label" style={{ marginTop: 4 }}>
              {scores.n_gold_matched}/{scores.n_gold} gold
              {scores.coverage_precision !== null &&
                ` · precision ${Math.round(scores.coverage_precision * 100)}%`}
            </div>
          </div>
        )}
      </div>
      {Object.keys(scores.hallucination_by_type).length > 0 && (
        <div style={{ marginTop: 14 }}>
          <div className="label" style={{ fontSize: 12, color: "var(--muted)", marginBottom: 6 }}>
            Hallucination by claim type
          </div>
          <table>
            <tbody>
              {Object.entries(scores.hallucination_by_type).map(([t, v]) => (
                <tr key={t}>
                  <td className="type-tag">{t}</td>
                  <td>{Math.round(v * 100)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      <div className="sub" style={{ marginTop: 10, marginBottom: 0 }}>{scores.coverage_note}</div>
    </div>
  );
}
