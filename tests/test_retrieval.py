import numpy as np

from src.indexing import build_bm25_index, build_tfidf_index
from src.retrievers import retrieve_biencoder, retrieve_bm25, retrieve_tfidf


def test_lexical_retrieval_descending_and_ids():
    candidate_ids = ["T1", "T2"]
    texts = ["powershell command", "file transfer"]
    query_ids = ["Q1"]
    queries = ["powershell"]
    vectorizer, matrix, _ = build_tfidf_index(
        texts,
        {"lowercase": True, "ngram_range": [1, 2], "min_df": 1, "sublinear_tf": True, "max_features": 50000},
    )
    rankings, scores, _ = retrieve_tfidf(vectorizer, matrix, queries, query_ids, candidate_ids)
    assert rankings["Q1"][0] == "T1"
    assert scores["Q1"][0] >= scores["Q1"][1]

    bm25, _, _ = build_bm25_index(texts)
    rankings, scores, _ = retrieve_bm25(bm25, queries, query_ids, candidate_ids)
    assert rankings["Q1"][0] == "T1"
    assert scores["Q1"][0] >= scores["Q1"][1]


def test_semantic_retrieval_helper():
    rankings, scores, _ = retrieve_biencoder(
        np.array([[1.0, 0.0]]),
        np.array([[0.9, 0.1], [0.0, 1.0]]),
        ["Q1"],
        ["T1", "T2"],
    )
    assert rankings["Q1"][0] == "T1"
    assert scores["Q1"][0] > scores["Q1"][1]
