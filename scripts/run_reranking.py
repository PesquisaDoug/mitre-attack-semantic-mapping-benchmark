from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sentence_transformers import CrossEncoder

from src.attention import crop_attention_matrix, extract_attention, load_attention_model, select_attention_example
from src.evaluation import ranking_rows
from src.reproducibility import commit_hash_from_config, now_utc, parameter_summary, write_json
from src.reranking import build_rerank_pairs, cross_encoder_scores, rerank_candidates
from src.visualization import plot_attention_matrix, plot_attention_top_tokens


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--profile", default="quick", choices=["quick", "full"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path.cwd()
    results_dir = root / "results"
    figures_dir = root / "figures"
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    model_config = yaml.safe_load((root / "configs" / "models.yaml").read_text(encoding="utf-8"))

    retrieval_path = results_dir / f"retrieval_rankings_{args.profile}.json"
    if not retrieval_path.exists():
        raise FileNotFoundError(f"Run retrieval first: {retrieval_path}")
    retrieval_payload = json.loads(retrieval_path.read_text(encoding="utf-8"))

    techniques = pd.read_csv(root / "data" / "techniques.csv")
    queries = pd.read_csv(root / "data" / "queries.csv")
    query_ids = queries["query_id"].tolist()
    query_texts = queries["query_text"].tolist()
    candidate_ids = techniques["attack_id"].tolist()
    candidate_texts = techniques["document_text"].tolist()
    technique_text_lookup = dict(zip(candidate_ids, candidate_texts))
    bi_rankings = retrieval_payload["methods"]["MiniLM-BiEncoder"]["rankings"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cross_config = model_config["transformers"]["cross_encoder"]
    top_k = int(config["reranking"]["top_k"])
    if top_k != int(cross_config["rerank_top_k"]):
        raise RuntimeError("Rerank top-k differs between experiment.yaml and models.yaml.")

    cross_encoder = CrossEncoder(cross_config["model_id"], device=device)
    pairs, metadata = build_rerank_pairs(query_ids, query_texts, bi_rankings, technique_text_lookup, top_k=top_k)
    batch_size = 64 if device == "cuda" else 16
    pair_scores, seconds = cross_encoder_scores(
        cross_encoder,
        pairs,
        batch_size=batch_size,
        show_progress_bar=args.profile != "full",
    )
    cross_rankings, cross_scores = rerank_candidates(query_ids, metadata, pair_scores)
    write_json(
        results_dir / f"reranking_rankings_{args.profile}.json",
        {
            "profile": args.profile,
            "method": "MiniLM-BiEncoder+CrossEncoder",
            "top_k": top_k,
            "pair_count": len(pairs),
            "rankings": cross_rankings,
            "scores": cross_scores,
        },
    )

    cross_rows = ranking_rows(
        techniques,
        queries,
        cross_rankings,
        cross_scores,
        "MiniLM-BiEncoder+CrossEncoder",
        max_rank=top_k,
    )
    strict_ids = set(queries.loc[queries["strict_eligible"], "query_id"])
    strict_top1 = cross_rows.loc[
        (cross_rows["query_id"].isin(strict_ids)) & (cross_rows["rank"] == 1),
        ["query_id", "attack_id", "technique_name", "score", "is_relevant"],
    ]
    failure_cases = (
        strict_top1.loc[~strict_top1["is_relevant"]]
        .merge(
            queries[
                [
                    "query_id",
                    "source_name",
                    "source_type",
                    "query_text",
                    "relevant_attack_ids",
                    "relevant_names",
                ]
            ],
            on="query_id",
            how="left",
        )
        .rename(
            columns={
                "attack_id": "predicted_attack_id",
                "technique_name": "predicted_technique_name",
                "score": "predicted_score",
            }
        )
    )
    failure_cases.to_csv(results_dir / "strict_crossencoder_failure_cases.csv", index=False)

    example = select_attention_example(queries, cross_rows, techniques)
    tokenizer, attention_model = load_attention_model(cross_config["model_id"], device)
    token_weights, attention_matrix, tokens = extract_attention(
        tokenizer,
        attention_model,
        example["query_text"],
        example["candidate_text"],
        device,
        max_length=int(config["attention"]["max_length"]),
    )
    token_weights.to_csv(results_dir / "attention_token_weights.csv", index=False)
    write_json(
        results_dir / "attention_example.json",
        {
            "query_id": example["query_id"],
            "query_text": example["query_text"],
            "top_candidate_attack_id": example["attack_id"],
            "top_candidate_name": example["candidate_name"],
            "top_candidate_relevant": bool(example["is_relevant"]),
            "analysis": (
                "Exploratory final-layer attention inspection. "
                "Attention weights are not interpreted as causal explanation."
            ),
        },
    )
    plot_attention_top_tokens(token_weights, figures_dir, int(config["attention"]["top_tokens"]))
    matrix, matrix_tokens = crop_attention_matrix(
        attention_matrix,
        tokens,
        int(config["attention"]["matrix_tokens"]),
    )
    plot_attention_matrix(matrix, matrix_tokens, figures_dir)

    cross_parameters = parameter_summary(cross_encoder.model)
    write_json(
        results_dir / f"reranking_state_{args.profile}.json",
        {
            "profile": args.profile,
            "completed_at_utc": now_utc(),
            "device": device,
            "pair_count": len(pairs),
            "top_k": top_k,
            "cross_encoder_seconds": seconds,
            "mean_rerank_latency_ms": seconds * 1000 / len(query_ids),
            "model_metadata": {
                "cross_encoder": {
                    "model_id": cross_config["model_id"],
                    **cross_parameters,
                    "huggingface_commit_hash": commit_hash_from_config(cross_encoder.model.config),
                    "rerank_top_k": top_k,
                }
            },
        },
    )
    print(json.dumps({"phase": "reranking", "profile": args.profile, "pairs": len(pairs)}, indent=2))


if __name__ == "__main__":
    main()
