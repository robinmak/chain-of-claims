"""Stage 5 — chunking.

Narrative text -> paragraph chunks. Tables -> BOTH row chunks (for context) and
cell chunks (for cell-level grounding and formula reconstruction). Cell-level
granularity is what lets Stage-6 grounding and Stage-8 arithmetic checks pin a claim
to a specific figure, per the Table-Text Alignment finding.
"""

from __future__ import annotations

from ..models import ChunkKind, EvidenceChunk
from .documents import ParsedBlock, ParsedTable


def chunk_blocks(blocks: list[ParsedBlock]) -> list[EvidenceChunk]:
    chunks: list[EvidenceChunk] = []
    for block in blocks:
        for pi, para in enumerate(block.paragraphs, start=1):
            chunks.append(
                EvidenceChunk(
                    source=block.source,
                    kind=ChunkKind.PARAGRAPH,
                    text=para,
                    locator=f"p{block.page}¶{pi}",
                )
            )
        for table in block.tables:
            chunks.extend(_chunk_table(block.source, block.page, table))
    return chunks


def _chunk_table(source: str, page: int, table: ParsedTable) -> list[EvidenceChunk]:
    out: list[EvidenceChunk] = []
    if not table.rows:
        return out
    header = table.rows[0]
    # Caption carries units/scale (e.g. "USD millions") that individual cells lack;
    # append it so Stage-8 can affirm "$500 million" from a cell reading "500".
    cap = f" [{table.caption}]" if table.caption else ""
    for ri, row in enumerate(table.rows):
        # Row chunk: the whole row as one line of context.
        row_text = " | ".join(row)
        out.append(
            EvidenceChunk(
                source=source,
                kind=ChunkKind.TABLE_ROW,
                text=f"{table.name}{cap} row {ri}: {row_text}",
                locator=f"{table.name} r{ri}",
            )
        )
        # Cell chunks: label each cell with its column header + row label for grounding.
        row_label = row[0] if row else ""
        for ci, cell in enumerate(row):
            if not cell:
                continue
            col = header[ci] if ci < len(header) else f"c{ci}"
            desc = f"{table.name}{cap}: {row_label} / {col} = {cell}"
            out.append(
                EvidenceChunk(
                    source=source,
                    kind=ChunkKind.TABLE_CELL,
                    text=desc,
                    locator=f"{table.name} r{ri}c{ci}",
                )
            )
    return out
