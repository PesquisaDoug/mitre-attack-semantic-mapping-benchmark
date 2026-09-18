from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def select_attention_example(
    queries: pd.DataFrame,
    rankings: pd.DataFrame,
    techniques: pd.DataFrame,
) -> dict[str, Any]:
    cross_rows = rankings.loc[rankings["method"] == "MiniLM-BiEncoder+CrossEncoder"].copy()
    top1 = cross_rows.loc[cross_rows["rank"] == 1][
        ["query_id", "attack_id", "technique_name", "is_relevant"]
    ]
    strict_ids = set(queries.loc[queries["strict_eligible"], "query_id"])
    correct_strict_top1 = top1.loc[top1["query_id"].isin(strict_ids) & top1["is_relevant"]]

    if not correct_strict_top1.empty:
        query_id = correct_strict_top1.iloc[0]["query_id"]
    else:
        strict_queries = queries.loc[queries["strict_eligible"]]
        query_id = queries.iloc[0]["query_id"] if strict_queries.empty else strict_queries.iloc[0]["query_id"]

    query = queries.set_index("query_id").loc[query_id]
    top_candidate = top1.set_index("query_id").loc[query_id]
    attack_id = top_candidate["attack_id"]
    technique = techniques.set_index("attack_id").loc[attack_id]
    return {
        "query_id": query_id,
        "query_text": str(query["query_text"]),
        "attack_id": attack_id,
        "candidate_text": str(technique["document_text"]),
        "candidate_name": str(technique["name"]),
        "is_relevant": bool(top_candidate["is_relevant"]),
    }


def load_attention_model(model_id: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            attn_implementation="eager",
        )
    except (TypeError, ValueError):
        model = AutoModelForSequenceClassification.from_pretrained(model_id)
    model = model.to(device)
    model.eval()
    return tokenizer, model


def extract_attention(
    tokenizer: Any,
    model: Any,
    query_text: str,
    candidate_text: str,
    device: str,
    max_length: int = 256,
) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    encoded = tokenizer(
        query_text,
        candidate_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    encoded_device = {key: value.to(device) for key, value in encoded.items()}
    with torch.inference_mode():
        outputs = model(**encoded_device, output_attentions=True, return_dict=True)
    if outputs.attentions is None:
        raise RuntimeError("The selected Cross-Encoder did not return attention tensors.")

    tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"][0])
    mean_attention = outputs.attentions[-1][0].detach().float().cpu().mean(dim=0)
    cls_attention = mean_attention[0].numpy()
    special_ids = set(tokenizer.all_special_ids)
    input_ids = encoded["input_ids"][0].tolist()
    token_rows = attention_token_rows(tokens, input_ids, cls_attention, special_ids)
    return pd.DataFrame(token_rows), mean_attention.numpy(), tokens


def attention_token_rows(
    tokens: list[str],
    token_ids: list[int],
    cls_attention: np.ndarray,
    special_ids: set[int],
) -> list[dict[str, Any]]:
    rows = []
    for position, (token, token_id, weight) in enumerate(zip(tokens, token_ids, cls_attention)):
        rows.append(
            {
                "position": position,
                "token": token,
                "token_id": int(token_id),
                "is_special": token_id in special_ids,
                "cls_attention": float(weight),
            }
        )
    return rows


def crop_attention_matrix(matrix: np.ndarray, tokens: list[str], max_tokens: int) -> tuple[np.ndarray, list[str]]:
    n = min(max_tokens, len(tokens))
    return matrix[:n, :n], tokens[:n]


def write_attention_outputs(
    results_dir: Path,
    example: dict[str, Any],
    token_weights: pd.DataFrame,
) -> None:
    token_weights.to_csv(results_dir / "attention_token_weights.csv", index=False)
    payload = {
        "query_id": example["query_id"],
        "query_text": example["query_text"],
        "top_candidate_attack_id": example["attack_id"],
        "top_candidate_name": example["candidate_name"],
        "top_candidate_relevant": bool(example["is_relevant"]),
        "analysis": (
            "Exploratory final-layer attention inspection. "
            "Attention weights are not interpreted as causal explanation."
        ),
    }
    (results_dir / "attention_example.json").write_text(
        __import__("json").dumps(payload, indent=2),
        encoding="utf-8",
    )
