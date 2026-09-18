from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Larger default font sizes so labels remain legible after IEEE two-column
# scaling (figures are typically rendered at ~3.4 inch column width).
plt.rcParams.update(
    {
        "font.size": 13,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
    }
)

METHOD_ORDER = ["TF-IDF", "BM25", "MiniLM-BiEncoder", "MiniLM-BiEncoder+CrossEncoder"]


def _reindex_methods(frame: pd.DataFrame) -> pd.DataFrame:
    existing_order = [method for method in METHOD_ORDER if method in frame.index]
    return frame.loc[existing_order]


def plot_technique_corpus(techniques: pd.DataFrame, figures_dir: Path) -> None:
    counts = pd.Series(
        {
            "Techniques": int((~techniques["is_subtechnique"]).sum()),
            "Sub-techniques": int(techniques["is_subtechnique"].sum()),
        }
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(counts.index, counts.values)
    ax.set_ylabel("Candidate documents")
    ax.set_title("Enterprise ATT&CK retrieval corpus")
    fig.tight_layout()
    fig.savefig(figures_dir / "01_technique_corpus.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_query_leakage(queries: pd.DataFrame, figures_dir: Path) -> None:
    counts = pd.Series(
        {
            "Strict eligible": int(queries["strict_eligible"].sum()),
            "Target ID present": int(queries["contains_target_id"].sum()),
            "Exact target name present": int(queries["contains_exact_target_name"].sum()),
        }
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(counts.index, counts.values)
    ax.set_ylabel("Queries")
    ax.set_title("Procedure-query lexical leakage audit")
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(figures_dir / "02_query_leakage_audit.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_query_source_types(source_type_counts: pd.DataFrame, figures_dir: Path) -> None:
    top_source_types = source_type_counts.head(10)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(top_source_types["source_type"], top_source_types["queries"])
    ax.set_ylabel("Queries")
    ax.set_title("Query source-object types")
    plt.xticks(rotation=25, ha="right")
    fig.tight_layout()
    fig.savefig(figures_dir / "03_query_source_types.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_strict_benchmark(metrics: pd.DataFrame, figures_dir: Path) -> None:
    strict_metrics = _reindex_methods(metrics.loc[metrics["subset"] == "strict"].set_index("method"))
    plot_columns = ["recall@1", "recall@5", "recall@10", "mrr@10", "ndcg@10"]
    short_labels = ["TF-IDF", "BM25", "BiEncoder", "BiEncoder\n+CrossEnc."]
    ax = strict_metrics[plot_columns].plot(kind="bar", figsize=(12, 5.5))
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Strict-subset ATT&CK semantic-mapping benchmark")
    ax.set_xticklabels(short_labels, rotation=0, ha="center")
    ax.legend(
        [column.replace("@", "@") for column in plot_columns],
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0,
    )
    plt.tight_layout()
    plt.savefig(figures_dir / "04_strict_retrieval_benchmark.png", dpi=220, bbox_inches="tight")
    plt.close()


def plot_all_vs_strict_ndcg(metrics: pd.DataFrame, figures_dir: Path) -> None:
    ndcg_pivot = _reindex_methods(metrics.pivot(index="method", columns="subset", values="ndcg@10"))
    ndcg_pivot = ndcg_pivot[["all", "strict"]]
    short_labels = ["TF-IDF", "BM25", "BiEncoder", "BiEncoder\n+CrossEnc."]
    ax = ndcg_pivot.plot(kind="bar", figsize=(9, 5.5))
    ax.set_ylabel("nDCG@10")
    ax.set_ylim(0, 1.05)
    ax.set_title("Effect of exact target-name/ID leakage filtering")
    ax.set_xticklabels(short_labels, rotation=0, ha="center")
    ax.legend(["All queries", "Strict queries"], loc="upper right")
    plt.tight_layout()
    plt.savefig(figures_dir / "05_all_vs_strict_ndcg.png", dpi=220, bbox_inches="tight")
    plt.close()


def plot_query_latency(efficiency: pd.DataFrame, figures_dir: Path) -> None:
    ordered = _reindex_methods(efficiency.set_index("method"))
    values = ordered["mean_query_latency_ms"]
    short_labels = ["TF-IDF", "BM25", "BiEncoder", "BiEncoder\n+CrossEnc."]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(range(len(values)), values.values, color="#4C72B0")
    ax.set_yscale("log")
    ax.set_ylabel("Mean latency (ms/query, log scale)")
    ax.set_title("Retrieval/reranking query latency (log scale)")
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(short_labels, rotation=0, ha="center")
    for bar, value in zip(bars, values.values):
        ax.annotate(
            f"{value:.3g}",
            xy=(bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    fig.tight_layout()
    fig.savefig(figures_dir / "06_query_latency.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_attention_top_tokens(token_weights: pd.DataFrame, figures_dir: Path, top_tokens: int) -> None:
    top = (
        token_weights.loc[~token_weights["is_special"]]
        .nlargest(top_tokens, "cls_attention")
        .sort_values("cls_attention")
    )
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(top["token"], top["cls_attention"])
    ax.set_xlabel("Mean final-layer [CLS] attention")
    ax.set_title("Exploratory Cross-Encoder token attention")
    fig.tight_layout()
    fig.savefig(figures_dir / "07_attention_top_tokens.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_attention_matrix(matrix: np.ndarray, tokens: list[str], figures_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 10))
    image = ax.imshow(matrix, aspect="auto")
    ax.set_xticks(np.arange(len(tokens)))
    ax.set_yticks(np.arange(len(tokens)))
    ax.set_xticklabels(tokens, rotation=90, fontsize=6)
    ax.set_yticklabels(tokens, fontsize=6)
    ax.set_xlabel("Attended-to token")
    ax.set_ylabel("Attending token")
    ax.set_title("Mean final-layer Cross-Encoder attention matrix")
    fig.colorbar(image, ax=ax, label="Attention weight")
    fig.tight_layout()
    fig.savefig(figures_dir / "08_attention_matrix.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
