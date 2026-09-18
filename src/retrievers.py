from __future__ import annotations

import time
from typing import Any

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.preprocessing import simple_tokenize


def _rank_scores(
    score_matrix: np.ndarray,
    query_ids: list[str],
    candidate_ids: list[str],
) -> tuple[dict[str, list[str]], dict[str, list[float]]]:
    candidate_id_array = np.asarray(candidate_ids)
    rankings: dict[str, list[str]] = {}
    scores: dict[str, list[float]] = {}
    for row_index, query_id in enumerate(query_ids):
        order = np.argsort(-score_matrix[row_index])
        rankings[query_id] = candidate_id_array[order].tolist()
        scores[query_id] = score_matrix[row_index, order].astype(float).tolist()
    return rankings, scores


def retrieve_tfidf(
    vectorizer: Any,
    tfidf_corpus: Any,
    query_texts: list[str],
    query_ids: list[str],
    candidate_ids: list[str],
) -> tuple[dict[str, list[str]], dict[str, list[float]], dict[str, float]]:
    start = time.perf_counter()
    query_matrix = vectorizer.transform(query_texts)
    similarity = cosine_similarity(query_matrix, tfidf_corpus)
    seconds = time.perf_counter() - start
    rankings, scores = _rank_scores(similarity, query_ids, candidate_ids)
    return rankings, scores, {"query_seconds": seconds, "similarity_seconds": seconds}


def retrieve_bm25(
    bm25: Any,
    query_texts: list[str],
    query_ids: list[str],
    candidate_ids: list[str],
) -> tuple[dict[str, list[str]], dict[str, list[float]], dict[str, float]]:
    candidate_id_array = np.asarray(candidate_ids)
    rankings: dict[str, list[str]] = {}
    scores_by_query: dict[str, list[float]] = {}
    start = time.perf_counter()
    for query_id, query_text in zip(query_ids, query_texts):
        scores = np.asarray(bm25.get_scores(simple_tokenize(query_text)), dtype=float)
        order = np.argsort(-scores)
        rankings[query_id] = candidate_id_array[order].tolist()
        scores_by_query[query_id] = scores[order].tolist()
    seconds = time.perf_counter() - start
    return rankings, scores_by_query, {"query_seconds": seconds}


def encode_queries(
    model: Any,
    query_texts: list[str],
    batch_size: int,
    normalize_embeddings: bool = True,
    show_progress_bar: bool = True,
) -> tuple[np.ndarray, float]:
    start = time.perf_counter()
    query_embeddings = model.encode(
        query_texts,
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
        convert_to_numpy=True,
        normalize_embeddings=normalize_embeddings,
    )
    return np.asarray(query_embeddings), time.perf_counter() - start


def retrieve_biencoder(
    query_embeddings: np.ndarray,
    candidate_embeddings: np.ndarray,
    query_ids: list[str],
    candidate_ids: list[str],
) -> tuple[dict[str, list[str]], dict[str, list[float]], dict[str, float]]:
    start = time.perf_counter()
    similarity = query_embeddings @ candidate_embeddings.T
    seconds = time.perf_counter() - start
    rankings, scores = _rank_scores(similarity, query_ids, candidate_ids)
    return rankings, scores, {"similarity_seconds": seconds}
