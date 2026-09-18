from __future__ import annotations

import json
import math
from typing import Any

import pandas as pd


def parse_relevant_ids(value: Any) -> set[str]:
    if isinstance(value, set):
        return set(map(str, value))
    if isinstance(value, list):
        return set(map(str, value))
    if isinstance(value, str):
        parsed = json.loads(value)
        return set(map(str, parsed))
    raise TypeError(f"Unsupported relevance value type: {type(value)}")


def metrics_for_ranking(
    ranking_ids: list[str],
    relevant_ids: set[str],
    cutoff: int = 10,
) -> dict[str, float]:
    if not relevant_ids:
        raise ValueError("At least one relevant document is required.")

    top = ranking_ids[:cutoff]
    result: dict[str, float] = {}

    for k in [1, 5, 10]:
        retrieved = ranking_ids[:k]
        relevant_retrieved = len(relevant_ids.intersection(retrieved))
        result[f"recall@{k}"] = relevant_retrieved / len(relevant_ids)
        result[f"hit@{k}"] = float(relevant_retrieved > 0)

    first_relevant_rank = None
    for rank, attack_id in enumerate(top, start=1):
        if attack_id in relevant_ids:
            first_relevant_rank = rank
            break

    result[f"mrr@{cutoff}"] = 1.0 / first_relevant_rank if first_relevant_rank is not None else 0.0

    dcg = 0.0
    precision_sum = 0.0
    relevant_seen = 0
    for rank, attack_id in enumerate(top, start=1):
        rel = int(attack_id in relevant_ids)
        if rel:
            relevant_seen += 1
            precision_sum += relevant_seen / rank
        dcg += rel / math.log2(rank + 1)

    ideal_relevant = min(len(relevant_ids), cutoff)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_relevant + 1))
    result[f"ndcg@{cutoff}"] = dcg / idcg if idcg > 0 else 0.0
    result[f"map@{cutoff}"] = precision_sum / ideal_relevant if ideal_relevant > 0 else 0.0
    return result


def evaluate_rankings(
    rankings: dict[str, list[str]],
    query_frame: pd.DataFrame,
    method_name: str,
    subset_name: str,
    cutoff: int = 10,
) -> tuple[pd.DataFrame, dict[str, float | int | str]]:
    rows = []
    query_lookup = query_frame.set_index("query_id")

    for query_id, ranking_ids in rankings.items():
        row = query_lookup.loc[query_id]
        relevant_ids = parse_relevant_ids(row["relevant_attack_ids"])
        metric_values = metrics_for_ranking(ranking_ids, relevant_ids, cutoff=cutoff)
        rows.append({"query_id": query_id, "method": method_name, "subset": subset_name, **metric_values})

    per_query = pd.DataFrame(rows)
    numeric_columns = [column for column in per_query.columns if column not in {"query_id", "method", "subset"}]
    aggregate: dict[str, float | int | str] = {
        "method": method_name,
        "subset": subset_name,
        "queries": len(per_query),
    }
    for column in numeric_columns:
        aggregate[column] = float(per_query[column].mean())
    return per_query, aggregate


def ranking_rows(
    techniques: pd.DataFrame,
    query_frame: pd.DataFrame,
    rankings: dict[str, list[str]],
    scores: dict[str, list[float]],
    method_name: str,
    max_rank: int = 20,
) -> pd.DataFrame:
    technique_lookup = techniques.set_index("attack_id")
    query_lookup = query_frame.set_index("query_id")
    rows = []

    for query_id, ranking_ids in rankings.items():
        relevant_ids = parse_relevant_ids(query_lookup.loc[query_id, "relevant_attack_ids"])
        method_scores = scores[query_id]
        for rank, (attack_id, score) in enumerate(
            zip(ranking_ids[:max_rank], method_scores[:max_rank]),
            start=1,
        ):
            technique = technique_lookup.loc[attack_id]
            rows.append(
                {
                    "query_id": query_id,
                    "method": method_name,
                    "rank": rank,
                    "attack_id": attack_id,
                    "technique_name": technique["name"],
                    "score": float(score),
                    "is_relevant": attack_id in relevant_ids,
                }
            )
    return pd.DataFrame(rows)
