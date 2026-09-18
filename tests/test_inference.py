import numpy as np

from src.reranking import rerank_candidates
from src.retrievers import retrieve_biencoder


def test_synthetic_inference_flow_returns_attack_id_and_numeric_score():
    rankings, scores, _ = retrieve_biencoder(
        np.array([[1.0, 0.0]]),
        np.array([[0.9, 0.1], [0.2, 0.8]]),
        ["Q1"],
        ["T1059.001", "T1105"],
    )
    assert rankings["Q1"][0] == "T1059.001"
    reranked, rerank_scores = rerank_candidates(
        ["Q1"],
        [("Q1", "T1059.001"), ("Q1", "T1105")],
        np.array([2.5, 0.1]),
    )
    assert reranked["Q1"][0].startswith("T")
    assert isinstance(rerank_scores["Q1"][0], float)
