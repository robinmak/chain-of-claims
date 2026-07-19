"""Stage-6 shortlist tests (deterministic, no LLM).

Locks in the grounding-recall fix: the true supporting table cell for a numeric claim
must survive shortlisting, even when it is a short cell competing with longer prose.
This is the failure mode that made the live fact-checker abstain on true figures.
"""

from chain_of_claims.models import ChunkKind, ClaimType, Claim, EvidenceChunk
from chain_of_claims.stages.s6_ground import _shortlist
from chain_of_claims.ingest.chunker import chunk_blocks
from chain_of_claims.ingest.documents import parse_document

from pathlib import Path

SAMPLES = Path(__file__).resolve().parents[1] / "samples"


def _cell(cid, text, loc):
    return EvidenceChunk(id=cid, source="src", kind=ChunkKind.TABLE_CELL, text=text, locator=loc)


def _para(cid, text, loc):
    return EvidenceChunk(id=cid, source="src", kind=ChunkKind.PARAGRAPH, text=text, locator=loc)


def test_true_cell_survives_shortlist_over_prose():
    claim = Claim(
        id=1,
        text="Acme's gross profit for Q3 FY2024 was $500 million.",
        type=ClaimType.EXTRACTED_METRIC,
    )
    # The one true cell, plus lots of longer distractor prose without the figure.
    true_cell = _cell(10, "income-p1-table0: Gross profit / Q3 FY2024 = 500", "t0 r3c1")
    distractors = [
        _para(20 + i,
              "The company discussed its quarterly performance and strategic outlook "
              "across multiple business segments during the earnings call in detail.",
              f"p1¶{i}")
        for i in range(20)
    ]
    chunks = distractors + [true_cell]
    shortlist = _shortlist(claim, chunks, k=8)
    assert any(c.id == 10 for c in shortlist), "true supporting cell must be shortlisted"


def test_number_normalisation_matches_dollar_and_plain():
    claim = Claim(id=2, text="Net income was $190 million.", type=ClaimType.EXTRACTED_METRIC)
    cell = _cell(30, "income: Net income = 190", "t0 r5c1")
    noise = [_para(40 + i, "General commentary with no figures here at all.", f"p2¶{i}")
             for i in range(15)]
    shortlist = _shortlist(claim, noise + [cell], k=5)
    assert any(c.id == 30 for c in shortlist), "$190 million must match plain 190 cell"


def test_table_cells_carry_unit_caption():
    """Cell chunks from the sample filing must carry the '(USD millions)' unit context
    from the table heading, so Stage-8 can affirm '$500 million' from a '500' cell."""
    blocks = parse_document(str(SAMPLES / "source_filing.md"))
    chunks = chunk_blocks(blocks)
    gross = [c for c in chunks if c.kind == ChunkKind.TABLE_CELL and "Gross profit" in c.text]
    assert gross, "expected a gross-profit cell chunk"
    assert any("USD millions" in c.text for c in gross), "cell must include unit caption"
