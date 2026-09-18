from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer

from src.preprocessing import simple_tokenize


def build_tfidf_index(candidate_texts: list[str], config: dict[str, Any]) -> tuple[TfidfVectorizer, Any, float]:
    vectorizer = TfidfVectorizer(
        lowercase=bool(config.get("lowercase", True)),
        ngram_range=tuple(config.get("ngram_range", [1, 2])),
        min_df=config.get("min_df", 1),
        sublinear_tf=bool(config.get("sublinear_tf", True)),
        max_features=config.get("max_features", 50000),
    )
    start = time.perf_counter()
    matrix = vectorizer.fit_transform(candidate_texts)
    return vectorizer, matrix, time.perf_counter() - start


def build_bm25_index(candidate_texts: list[str]) -> tuple[BM25Okapi, list[list[str]], float]:
    corpus_tokens = [simple_tokenize(text) for text in candidate_texts]
    start = time.perf_counter()
    index = BM25Okapi(corpus_tokens)
    return index, corpus_tokens, time.perf_counter() - start


def save_index(index: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(index, path)


def load_index(path: Path) -> object:
    return joblib.load(path)


def encode_candidate_corpus(
    model: Any,
    candidate_texts: list[str],
    batch_size: int,
    normalize_embeddings: bool = True,
    show_progress_bar: bool = True,
) -> tuple[np.ndarray, float]:
    start = time.perf_counter()
    embeddings = model.encode(
        candidate_texts,
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
        convert_to_numpy=True,
        normalize_embeddings=normalize_embeddings,
    )
    return np.asarray(embeddings), time.perf_counter() - start


def save_candidate_embeddings(embeddings: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, embeddings)


def load_candidate_embeddings(path: Path) -> np.ndarray:
    return np.load(path)


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms
