import { ClaimView } from "./types";

// Self-contained SVG rendering of one claim's Toulmin + evidence structure:
//   Reason  --warrant-->  Claim  -->  Evidence chunks
// The claim node is colored by fact-check verdict; edges to evidence are drawn
// per grounded chunk. No external graph library (keeps the bundle small).

const VERDICT_COLOR: Record<string, string> = {
  supported: "#3fb950",
  refuted: "#f85149",
  not_enough_evidence: "#d29922",
  not_checked: "#9aa4b2",
};

function wrap(text: string, max: number, maxLines: number): string[] {
  const words = text.split(/\s+/);
  const lines: string[] = [];
  let cur = "";
  for (const w of words) {
    if ((cur + " " + w).trim().length > max) {
      lines.push(cur.trim());
      cur = w;
      if (lines.length === maxLines - 1) break;
    } else {
      cur = (cur + " " + w).trim();
    }
  }
  if (cur && lines.length < maxLines) lines.push(cur.trim());
  const joined = lines.join(" ");
  if (joined.length < text.length) lines[lines.length - 1] += "…";
  return lines;
}

function Node({
  x, y, w, h, title, lines, stroke, fill,
}: {
  x: number; y: number; w: number; h: number;
  title: string; lines: string[]; stroke: string; fill: string;
}) {
  return (
    <g>
      <rect x={x} y={y} width={w} height={h} rx={8} fill={fill} stroke={stroke} strokeWidth={1.5} />
      <text x={x + 10} y={y + 18} fontSize={11} fontWeight={700} fill={stroke}>
        {title}
      </text>
      {lines.map((ln, i) => (
        <text key={i} x={x + 10} y={y + 36 + i * 14} fontSize={11} fill="#e6e9ef">
          {ln}
        </text>
      ))}
    </g>
  );
}

export function ClaimEvidenceGraph({ c }: { c: ClaimView }) {
  const verdict = c.verdict?.factcheck_result || "not_checked";
  const claimColor = VERDICT_COLOR[verdict] || "#9aa4b2";

  const colW = 230;
  const gapX = 70;
  const reasonX = 10;
  const claimX = reasonX + colW + gapX;
  const evX = claimX + colW + gapX;

  const evidence = c.evidence.slice(0, 6);
  const evH = 40;
  const evGap = 12;
  const evTotal = Math.max(1, evidence.length) * (evH + evGap);
  const height = Math.max(180, evTotal + 40);
  const midY = height / 2;

  const reasonY = midY - 45;
  const claimY = midY - 45;
  const claimNodeH = 90;
  const reasonH = 80;

  const claimCenterY = claimY + claimNodeH / 2;
  const claimRightX = claimX + colW;

  const reasonLines = wrap(c.triplet?.reason || "(no grounds stated)", 34, 3);
  const claimLines = wrap(c.text, 34, 3);
  const warrant = c.triplet?.warrant || "";
  const warrantLines = wrap(warrant, 46, 2);

  return (
    <div className="detail" style={{ overflowX: "auto" }}>
      <h4 style={{ marginTop: 0 }}>Claim–evidence graph</h4>
      <svg width={evX + colW + 20} height={height} style={{ minWidth: evX + colW + 20 }}>
        {/* Reason -> Claim edge, labeled with the warrant */}
        <defs>
          <marker id="arrow" markerWidth="8" markerHeight="8" refX="6" refY="3"
                  orient="auto" markerUnits="strokeWidth">
            <path d="M0,0 L6,3 L0,6 Z" fill="#5b9dff" />
          </marker>
        </defs>
        <line
          x1={reasonX + colW} y1={reasonY + reasonH / 2}
          x2={claimX} y2={claimCenterY}
          stroke="#5b9dff" strokeWidth={1.5} markerEnd="url(#arrow)"
        />
        <text
          x={(reasonX + colW + claimX) / 2} y={reasonY + reasonH / 2 - 8}
          fontSize={10} fill="#9aa4b2" textAnchor="middle"
        >
          warrant{c.triplet?.is_causal ? " (causal)" : ""}
        </text>
        {warrantLines.map((ln, i) => (
          <text
            key={i}
            x={(reasonX + colW + claimX) / 2} y={reasonY + reasonH / 2 + 8 + i * 12}
            fontSize={9} fill="#9aa4b2" textAnchor="middle" fontStyle="italic"
          >
            {ln}
          </text>
        ))}

        {/* Claim -> each evidence chunk */}
        {evidence.map((_, i) => {
          const ey = 20 + i * (evH + evGap) + evH / 2;
          return (
            <line
              key={i}
              x1={claimRightX} y1={claimCenterY}
              x2={evX} y2={ey}
              stroke={claimColor} strokeWidth={1.2} opacity={0.7} markerEnd="url(#arrow)"
            />
          );
        })}

        {/* Nodes */}
        <Node
          x={reasonX} y={reasonY} w={colW} h={reasonH}
          title="REASON" lines={reasonLines} stroke="#9aa4b2" fill="#1f232c"
        />
        <Node
          x={claimX} y={claimY} w={colW} h={claimNodeH}
          title={`CLAIM · ${verdict}`} lines={claimLines} stroke={claimColor} fill="#1f232c"
        />
        {evidence.map((e, i) => {
          const ey = 20 + i * (evH + evGap);
          return (
            <Node
              key={i}
              x={evX} y={ey} w={colW} h={evH}
              title={`${e.source} [${e.locator}]`}
              lines={wrap(e.text, 36, 1)}
              stroke="#5b9dff" fill="#181b22"
            />
          );
        })}
        {evidence.length === 0 && (
          <text x={evX + 10} y={midY} fontSize={11} fill="#9aa4b2">
            No grounding evidence
          </text>
        )}
      </svg>
    </div>
  );
}
