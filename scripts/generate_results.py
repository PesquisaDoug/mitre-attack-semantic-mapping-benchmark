from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd
import yaml

from src.evaluation import evaluate_rankings, ranking_rows
from src.reproducibility import device_metadata, get_git_commit, now_utc, package_versions, write_json
from src.visualization import plot_all_vs_strict_ndcg, plot_query_latency, plot_strict_benchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--profile", default="quick", choices=["quick", "full"])
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    root = Path.cwd()
    data_dir = root / "data"
    results_dir = root / "results"
    figures_dir = root / "figures"
    paper_figures_dir = root / "paper" / "figures"
    paper_figures_dir.mkdir(parents=True, exist_ok=True)

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    report_cutoff = int(config["evaluation"].get("report_cutoff", 10))
    top_k = int(config["reranking"]["top_k"])

    techniques = pd.read_csv(data_dir / "techniques.csv")
    queries = pd.read_csv(data_dir / "queries.csv")
    data_manifest = load_json(data_dir / "data_manifest.json")
    retrieval_payload = load_json(results_dir / f"retrieval_rankings_{args.profile}.json")
    reranking_payload = load_json(results_dir / f"reranking_rankings_{args.profile}.json")
    retrieval_state = load_json(results_dir / f"retrieval_state_{args.profile}.json")
    reranking_state = load_json(results_dir / f"reranking_state_{args.profile}.json")

    ranking_methods = {
        "TF-IDF": retrieval_payload["methods"]["TF-IDF"],
        "BM25": retrieval_payload["methods"]["BM25"],
        "MiniLM-BiEncoder": retrieval_payload["methods"]["MiniLM-BiEncoder"],
        "MiniLM-BiEncoder+CrossEncoder": {
            "rankings": reranking_payload["rankings"],
            "scores": reranking_payload["scores"],
        },
    }

    aggregate_rows = []
    per_query_frames = []
    ranking_frames = []
    subsets = {
        "all": queries,
        "strict": queries.loc[queries["strict_eligible"]].copy(),
    }

    for subset_name, subset_df in subsets.items():
        subset_query_ids = set(subset_df["query_id"])
        for method_name, method_payload in ranking_methods.items():
            method_rankings = method_payload["rankings"]
            method_scores = method_payload["scores"]
            subset_rankings = {
                query_id: ranking
                for query_id, ranking in method_rankings.items()
                if query_id in subset_query_ids
            }
            subset_scores = {query_id: method_scores[query_id] for query_id in subset_rankings}
            per_query, aggregate = evaluate_rankings(
                subset_rankings,
                subset_df,
                method_name,
                subset_name,
                cutoff=report_cutoff,
            )
            per_query_frames.append(per_query)
            aggregate_rows.append(aggregate)
            if subset_name == "all":
                ranking_frames.append(
                    ranking_rows(
                        techniques,
                        queries,
                        subset_rankings,
                        subset_scores,
                        method_name,
                        max_rank=top_k,
                    )
                )

    metrics = pd.DataFrame(aggregate_rows)
    per_query_metrics = pd.concat(per_query_frames, ignore_index=True)
    rankings = pd.concat(ranking_frames, ignore_index=True)
    metrics.to_csv(results_dir / "metrics.csv", index=False)
    per_query_metrics.to_csv(results_dir / "per_query_metrics.csv", index=False)
    rankings.to_csv(results_dir / "rankings.csv", index=False)

    bi_eff = retrieval_state["efficiency"]["MiniLM-BiEncoder"]
    cross_eff = reranking_state
    model_metadata = {
        **retrieval_state["model_metadata"],
        **reranking_state["model_metadata"],
    }
    combined_parameter_count = int(
        model_metadata["bi_encoder"]["parameter_count"]
        + model_metadata["cross_encoder"]["parameter_count"]
    )
    combined_parameter_memory_mb = float(
        model_metadata["bi_encoder"]["parameter_memory_mb"]
        + model_metadata["cross_encoder"]["parameter_memory_mb"]
    )
    embedding_artifact_mb = bi_eff["local_artifact_mb"]
    efficiency = pd.DataFrame(
        [
            {"method": "TF-IDF", **retrieval_state["efficiency"]["TF-IDF"]},
            {"method": "BM25", **retrieval_state["efficiency"]["BM25"]},
            {"method": "MiniLM-BiEncoder", **bi_eff},
            {
                "method": "MiniLM-BiEncoder+CrossEncoder",
                "index_seconds": bi_eff["index_seconds"],
                "mean_query_latency_ms": bi_eff["mean_query_latency_ms"] + cross_eff["mean_rerank_latency_ms"],
                "parameter_count": combined_parameter_count,
                "parameter_memory_mb": combined_parameter_memory_mb,
                "local_artifact_mb": embedding_artifact_mb,
            },
        ]
    )
    efficiency.to_csv(results_dir / "efficiency.csv", index=False)
    write_json(results_dir / "model_metadata.json", model_metadata)

    strict_lookup = metrics.loc[metrics["subset"] == "strict"].set_index("method")
    gain_rows = []
    for metric in ["recall@1", "mrr@10", "ndcg@10"]:
        bi_value = float(strict_lookup.loc["MiniLM-BiEncoder", metric])
        cross_value = float(strict_lookup.loc["MiniLM-BiEncoder+CrossEncoder", metric])
        gain_rows.append(
            {
                "metric": metric,
                "bi_encoder": bi_value,
                "reranked": cross_value,
                "absolute_gain": cross_value - bi_value,
            }
        )
    pd.DataFrame(gain_rows).to_csv(results_dir / "reranking_gain.csv", index=False)

    plot_strict_benchmark(metrics, figures_dir)
    plot_all_vs_strict_ndcg(metrics, figures_dir)
    plot_query_latency(efficiency, figures_dir)
    for figure in figures_dir.glob("*.png"):
        shutil.copy2(figure, paper_figures_dir / figure.name)

    experiment_manifest = {
        "experiment_name": "mitre_attack_transformer_semantic_mapping",
        "title": (
            "Transformer-Based Semantic Mapping of Cyber Threat Intelligence to "
            "MITRE ATT&CK Techniques: A Retrieval and Reranking Benchmark"
        ),
        "seed": config["seed"],
        "profile": args.profile,
        "quick_mode": args.profile == "quick",
        "max_queries": config["profiles"][args.profile]["max_queries"],
        "rerank_top_k": top_k,
        "report_cutoff": report_cutoff,
        "dataset": data_manifest,
        "models": model_metadata,
        "query_subsets": {
            "all": int(len(queries)),
            "strict": int(queries["strict_eligible"].sum()),
        },
        "metric_definitions": {
            "recall@k": "fraction of relevant ATT&CK techniques retrieved in top-k",
            "hit@k": "whether at least one relevant ATT&CK technique occurs in top-k",
            "mrr@10": "reciprocal rank of the first relevant document, cutoff 10",
            "ndcg@10": "binary relevance normalized discounted cumulative gain, cutoff 10",
            "map@10": "mean average precision using binary relevance, cutoff 10",
        },
        "efficiency": efficiency.to_dict(orient="records"),
        "device": device_metadata(),
        "software": package_versions(),
        "git_commit": get_git_commit(root),
        "completed_at_utc": now_utc(),
    }
    write_json(results_dir / "experiment_manifest.json", experiment_manifest)

    paper_summary = {
        "title": experiment_manifest["title"],
        "research_questions": {
            "RQ1": "Lexical versus pretrained semantic retrieval.",
            "RQ2": "Cross-Encoder reranking gain over bi-encoder retrieval.",
            "RQ3": "Performance after exact target-name/ID leakage filtering.",
            "RQ4": "Exploratory Cross-Encoder attention inspection.",
            "RQ5": "Quality/latency/model-footprint trade-offs.",
        },
        "main_results": {
            "metrics": "results/metrics.csv",
            "per_query_metrics": "results/per_query_metrics.csv",
            "rankings": "results/rankings.csv",
            "efficiency": "results/efficiency.csv",
            "reranking_gain": "results/reranking_gain.csv",
            "attention": "results/attention_example.json",
            "failure_cases": "results/strict_crossencoder_failure_cases.csv",
        },
        "figures_directory": "figures/",
    }
    write_json(results_dir / "paper_summary.json", paper_summary)

    required_outputs = [
        data_dir / "data_manifest.json",
        data_dir / "techniques.csv",
        data_dir / "queries.csv",
        results_dir / "data_quality_summary.csv",
        results_dir / "query_source_types.csv",
        results_dir / "metrics.csv",
        results_dir / "per_query_metrics.csv",
        results_dir / "rankings.csv",
        results_dir / "efficiency.csv",
        results_dir / "reranking_gain.csv",
        results_dir / "model_metadata.json",
        results_dir / "attention_token_weights.csv",
        results_dir / "attention_example.json",
        results_dir / "strict_crossencoder_failure_cases.csv",
        results_dir / "experiment_manifest.json",
        results_dir / "paper_summary.json",
        root / "artifacts" / "models" / "tfidf_vectorizer.joblib",
        root / "artifacts" / "models" / "bm25_index.joblib",
        root / "artifacts" / "embeddings" / "technique_embeddings.npy",
        root / "app" / "streamlit_app.py",
    ]
    missing = [str(path) for path in required_outputs if not path.exists()]
    if missing:
        raise FileNotFoundError("Expected artifacts were not created:\n" + "\n".join(missing))

    print(json.dumps({"phase": "generate_results", "profile": args.profile, "metrics_rows": len(metrics)}, indent=2))


if __name__ == "__main__":
    main()
