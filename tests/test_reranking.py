import numpy as np

from src.reranking import build_rerank_pairs, rerank_candidates, rerank_with_mock_scores


def test_pair_construction_uses_only_top_k():
    pairs, metadata = build_rerank_pairs(
        ["Q1"],
        ["query"],
        {"Q1": ["T1", "T2", "T3"]},
        {"T1": "doc1", "T2": "doc2", "T3": "doc3"},
        top_k=2,
    )
    assert pairs == [("query", "doc1"), ("query", "doc2")]
    assert metadata == [("Q1", "T1"), ("Q1", "T2")]


def test_rerank_candidates_reorders_without_new_candidates():
    rankings, scores = rerank_candidates(
        ["Q1"],
        [("Q1", "T1"), ("Q1", "T2")],
        np.array([0.1, 0.9]),
    )
    assert rankings["Q1"] == ["T2", "T1"]
    assert scores["Q1"] == [0.9, 0.1]
    assert rerank_with_mock_scores("Q1", ["T1", "T2"], [0.1, 0.9]) == ["T2", "T1"]
