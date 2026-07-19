"""Document ingestion: PDF, plain text, and markdown.

Produces a normalized intermediate: a list of `ParsedBlock`s that the chunker turns
into evidence chunks. Tables are kept structured (rows of cells) so the chunker can
emit cell/row-level evidence, which Stage-6 grounding and Stage-8 formula checks need.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ParsedTable:
    name: str                       # e.g. "p3-table1"
    rows: list[list[str]]           # rows of cell strings (row 0 often headers)
    caption: str = ""               # nearest preceding heading/label (carries units/scale)


@dataclass
class ParsedBlock:
    source: str
    page: int
    paragraphs: list[str] = field(default_factory=list)
    tables: list[ParsedTable] = field(default_factory=list)


def parse_document(path: str | Path) -> list[ParsedBlock]:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".pdf":
        return _parse_pdf(p)
    if suffix in (".txt", ".md", ".markdown", ".text"):
        return _parse_text(p)
    raise ValueError(f"Unsupported document type: {suffix} ({p.name})")


def _parse_text(p: Path) -> list[ParsedBlock]:
    text = p.read_text(encoding="utf-8", errors="replace")
    paragraphs = [para.strip() for para in text.split("\n\n") if para.strip()]
    tables: list[ParsedTable] = []
    # Detect simple markdown/pipe tables and pull them out of the paragraph stream.
    kept_paras: list[str] = []
    table_idx = 0
    buffer: list[str] = []
    last_para = ""  # nearest preceding non-table paragraph -> table caption (units/scale)

    def flush_table():
        nonlocal table_idx, buffer
        if len(buffer) >= 2:
            rows = [
                [c.strip() for c in line.strip().strip("|").split("|")]
                for line in buffer
                # skip markdown separator rows like |---|---|
                if not set(line.replace("|", "").strip()) <= set("-: ")
            ]
            rows = [r for r in rows if any(r)]
            if rows:
                tables.append(
                    ParsedTable(
                        name=f"{p.stem}-table{table_idx}",
                        rows=rows,
                        caption=last_para.lstrip("# ").strip(),
                    )
                )
                table_idx += 1
        buffer = []

    for para in paragraphs:
        lines = para.splitlines()
        if lines and all(line.strip().startswith("|") for line in lines):
            buffer.extend(lines)
            flush_table()
        else:
            kept_paras.append(para)
            last_para = para

    return [ParsedBlock(source=p.name, page=1, paragraphs=kept_paras, tables=tables)]


def _parse_pdf(p: Path) -> list[ParsedBlock]:
    import pdfplumber

    blocks: list[ParsedBlock] = []
    with pdfplumber.open(p) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            paragraphs = [para.strip() for para in text.split("\n\n") if para.strip()]
            tables: list[ParsedTable] = []
            # Heuristic caption: the last text paragraph on the page usually precedes
            # or labels the table(s) and often carries the unit/scale (e.g. "USD millions").
            caption = paragraphs[-1].strip() if paragraphs else ""
            for t_idx, tbl in enumerate(page.extract_tables() or []):
                rows = [[(c or "").strip() for c in row] for row in tbl]
                rows = [r for r in rows if any(r)]
                if rows:
                    tables.append(
                        ParsedTable(
                            name=f"{p.stem}-p{i}-table{t_idx}", rows=rows, caption=caption
                        )
                    )
            blocks.append(
                ParsedBlock(source=p.name, page=i, paragraphs=paragraphs, tables=tables)
            )
    return blocks
