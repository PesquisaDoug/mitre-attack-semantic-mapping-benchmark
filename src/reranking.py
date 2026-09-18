from __future__ import annotations

import time
from collections import defaultdict
from typing import Any

import numpy as np


def build_rerank_pairs(
    query_ids: list[str],
    query_texts: list[str],
    bi_rankings: dict[str, list[str]],
    technique_text_lookup: dict[str, str],
    top_k: int = 20,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    pairs: list[tuple[str, str]] = []
    metadata: list[tuple[str, str]] = []
    for query_id, query_text in zip(query_ids, query_texts):
        candidate_top_ids = bi_rankings[query_id][:top_k]
        for attack_id in candidate_top_ids:
            pairs.append((query_text, technique_text_lookup[attack_id]))
            metadata.append((query_id, attack_id))
    return pairs, metadata


def cross_encoder_scores(
    cross_encoder: Any,
    pairs: list[tuple[str, str]],
    batch_size: int,
    show_progress_bar: bool = True,
) -> tuple[np.ndarray, float]:
    start = time.perf_counter()
    scores = cross_encoder.predict(
        pairs,
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
    )
    return np.asarray(scores).reshape(-1), time.perf_counter() - start


def rerank_candidates(
    query_ids: list[str],
    metadata: list[tuple[str, str]],
    pair_scores: np.ndarray,
) -> tuple[dict[str, list[str]], dict[str, list[float]]]:
    grouped: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for (query_id, attack_id), score in zip(metadata, pair_scores):
        grouped[query_id].append((attack_id, float(score)))

    rankings: dict[str, list[str]] = {}
    scores: dict[str, list[float]] = {}
    for query_id in query_ids:
        ranked = sorted(grouped[query_id], key=lambda item: item[1], reverse=True)
        rankings[query_id] = [item[0] for item in ranked]
        scores[query_id] = [item[1] for item in ranked]
    return rankings, scores


def rerank_with_mock_scores(
    query_id: str,
    candidates: list[str],
    scores: list[float],
) -> list[str]:
    return [candidate for candidate, _ in sorted(zip(candidates, scores), key=lambda item: item[1], reverse=True)]
