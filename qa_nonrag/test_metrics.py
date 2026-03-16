from __future__ import annotations

from .pipeline import citations_pr, parse_citations, token_f1


def test_parse_citations():
    text = "Resposta baseada em [DOC_A#p0] e também [DOC_B#p2]."
    citations = parse_citations(text)
    assert citations == ["DOC_A#p0", "DOC_B#p2"]


def test_citations_pr():
    precision, recall = citations_pr(
        pred_citations=["DOC_A#p0", "DOC_B#p2"],
        gold_citations=["DOC_A#p0"],
    )
    assert abs(precision - 0.5) < 1e-9
    assert abs(recall - 1.0) < 1e-9


def test_token_f1_identity():
    score = token_f1("total_duration e load_duration", "total_duration e load_duration")
    assert score == 1.0