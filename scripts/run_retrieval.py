from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sentence_transformers import SentenceTransformer

from src.data import (
    build_procedure_queries,
    build_source_index,
    build_technique_corpus,
    create_data_manifest,
    data_quality_summary,
    ensure_stix_bundle,
    load_stix_bundle,
    query_source_type_counts,
    sample_queries,
)
from src.indexing import (
    build_bm25_index,
    build_tfidf_index,
    encode_candidate_corpus,
    save_candidate_embeddings,
    save_index,
)
from src.reproducibility import commit_hash_from_config, now_utc, parameter_summary, write_json
from src.retrievers import encode_queries, retrieve_biencoder, retrieve_bm25, retrieve_tfidf
from src.visualization import plot_query_leakage, plot_query_source_types, plot_technique_corpus


def truncate_rankings(
    rankings: dict[str, list[str]],
    scores: dict[str, list[float]],
    top_k: int,
) -> dict[str, dict[str, list[str] | list[float]]]:
    return {
        "rankings": {query_id: values[:top_k] for query_id, values in rankings.items()},
        "scores": {query_id: values[:top_k] for query_id, values in scores.items()},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--profile", default="quick", choices=["quick", "full"])
    return parser.parse_args()


def mkdirs(root: Path) -> dict[str, Path]:
    paths = {
        "data": root / "data",
        "raw": root / "data" / "raw",
        "results": root / "results",
        "figures": root / "figures",
        "models": root / "artifacts" / "models",
        "embeddings": root / "artifacts" / "embeddings",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def main() -> None:
    args = parse_args()
    root = Path.cwd()
    paths = mkdirs(root)

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    model_config = yaml.safe_load((root / "configs" / "models.yaml").read_text(encoding="utf-8"))
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    max_queries = config["profiles"][args.profile]["max_queries"]
    dataset_config = config["dataset"]
    bundle_path = paths["raw"] / dataset_config.get("bundle_file", "enterprise-attack.json")
    ensure_stix_bundle(dataset_config["bundle_url"], bundle_path)
    _, objects = load_stix_bundle(bundle_path)

    techniques, technique_by_stix_id = build_technique_corpus(objects)
    object_by_id = build_source_index(objects)
    all_queries = build_procedure_queries(
        objects,
        technique_by_stix_id,
        object_by_id,
        relationship_type=config["query_construction"]["relationship_type"],
    )
    queries = sample_queries(all_queries, max_queries=max_queries, seed=seed)

    techniques.to_csv(paths["data"] / "techniques.csv", index=False)
    queries.to_csv(paths["data"] / "queries.csv", index=False)
    data_quality_summary(techniques, all_queries, queries).to_csv(
        paths["results"] / "data_quality_summary.csv",
        index=False,
    )
    source_counts = query_source_type_counts(queries)
    source_counts.to_csv(paths["results"] / "query_source_types.csv", index=False)
    manifest = create_data_manifest(
        objects=objects,
        bundle_path=bundle_path,
        bundle_url=dataset_config["bundle_url"],
        techniques=techniques,
        all_queries=all_queries,
        queries=queries,
        canonical_repository=dataset_config["source_repository"],
        domain=dataset_config["domain"],
    )
    write_json(paths["data"] / "data_manifest.json", manifest)

    plot_technique_corpus(techniques, paths["figures"])
    plot_query_leakage(queries, paths["figures"])
    plot_query_source_types(source_counts, paths["figures"])

    candidate_ids = techniques["attack_id"].tolist()
    candidate_texts = techniques["document_text"].tolist()
    query_ids = queries["query_id"].tolist()
    query_texts = queries["query_text"].tolist()

    tfidf_config = model_config["lexical"]["tfidf"]
    tfidf_vectorizer, tfidf_corpus, tfidf_index_seconds = build_tfidf_index(candidate_texts, tfidf_config)
    save_index(tfidf_vectorizer, paths["models"] / "tfidf_vectorizer.joblib")
    tfidf_rankings, tfidf_scores, tfidf_timing = retrieve_tfidf(
        tfidf_vectorizer,
        tfidf_corpus,
        query_texts,
        query_ids,
        candidate_ids,
    )

    bm25, _, bm25_index_seconds = build_bm25_index(candidate_texts)
    save_index(bm25, paths["models"] / "bm25_index.joblib")
    bm25_rankings, bm25_scores, bm25_timing = retrieve_bm25(
        bm25,
        query_texts,
        query_ids,
        candidate_ids,
    )

    bi_config = model_config["transformers"]["bi_encoder"]
    batch_size = 128 if device == "cuda" else 32
    bi_encoder = SentenceTransformer(bi_config["model_id"], device=device)
    embedding_dimension = bi_encoder.get_sentence_embedding_dimension()
    if embedding_dimension != 384:
        raise RuntimeError(f"Expected MiniLM embedding dimension 384, got {embedding_dimension}.")
    candidate_embeddings, bi_index_seconds = encode_candidate_corpus(
        bi_encoder,
        candidate_texts,
        batch_size=batch_size,
        normalize_embeddings=bool(bi_config.get("normalize_embeddings", True)),
    )
    save_candidate_embeddings(candidate_embeddings, paths["embeddings"] / "technique_embeddings.npy")
    query_embeddings, bi_query_encoding_seconds = encode_queries(
        bi_encoder,
        query_texts,
        batch_size=batch_size,
        normalize_embeddings=bool(bi_config.get("normalize_embeddings", True)),
    )
    bi_rankings, bi_scores, bi_timing = retrieve_biencoder(
        query_embeddings,
        candidate_embeddings,
        query_ids,
        candidate_ids,
    )

    bi_module = bi_encoder._first_module().auto_model
    bi_parameters = parameter_summary(bi_module)
    saved_top_k = max(int(config["reranking"]["top_k"]), int(config["evaluation"].get("report_cutoff", 10)))
    retrieval_payload = {
        "profile": args.profile,
        "saved_top_k": saved_top_k,
        "query_ids": query_ids,
        "candidate_ids": candidate_ids,
        "methods": {
            "TF-IDF": truncate_rankings(tfidf_rankings, tfidf_scores, saved_top_k),
            "BM25": truncate_rankings(bm25_rankings, bm25_scores, saved_top_k),
            "MiniLM-BiEncoder": truncate_rankings(bi_rankings, bi_scores, saved_top_k),
        },
    }
    write_json(paths["results"] / f"retrieval_rankings_{args.profile}.json", retrieval_payload)

    tfidf_artifact = paths["models"] / "tfidf_vectorizer.joblib"
    bm25_artifact = paths["models"] / "bm25_index.joblib"
    embedding_artifact = paths["embeddings"] / "technique_embeddings.npy"
    retrieval_state = {
        "profile": args.profile,
        "completed_at_utc": now_utc(),
        "device": device,
        "query_count": len(query_ids),
        "candidate_count": len(candidate_ids),
        "tfidf_vocabulary_size": len(tfidf_vectorizer.vocabulary_),
        "candidate_embedding_shape": list(candidate_embeddings.shape),
        "efficiency": {
            "TF-IDF": {
                "index_seconds": tfidf_index_seconds,
                "mean_query_latency_ms": tfidf_timing["query_seconds"] * 1000 / len(query_ids),
                "parameter_count": 0,
                "parameter_memory_mb": 0.0,
                "local_artifact_mb": tfidf_artifact.stat().st_size / (1024 ** 2),
            },
            "BM25": {
                "index_seconds": bm25_index_seconds,
                "mean_query_latency_ms": bm25_timing["query_seconds"] * 1000 / len(query_ids),
                "parameter_count": 0,
                "parameter_memory_mb": 0.0,
                "local_artifact_mb": bm25_artifact.stat().st_size / (1024 ** 2),
            },
            "MiniLM-BiEncoder": {
                "index_seconds": bi_index_seconds,
                "query_encoding_seconds": bi_query_encoding_seconds,
                "similarity_seconds": bi_timing["similarity_seconds"],
                "mean_query_latency_ms": (
                    bi_query_encoding_seconds + bi_timing["similarity_seconds"]
                )
                * 1000
                / len(query_ids),
                **bi_parameters,
                "local_artifact_mb": embedding_artifact.stat().st_size / (1024 ** 2),
            },
        },
        "model_metadata": {
            "bi_encoder": {
                "model_id": bi_config["model_id"],
                "embedding_dimension": embedding_dimension,
                **bi_parameters,
                "huggingface_commit_hash": commit_hash_from_config(bi_module.config),
            }
        },
    }
    write_json(paths["results"] / f"retrieval_state_{args.profile}.json", retrieval_state)
    print(json.dumps({"phase": "retrieval", "profile": args.profile, "queries": len(query_ids)}, indent=2))


if __name__ == "__main__":
    main()
