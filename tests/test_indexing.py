import numpy as np

from src.indexing import build_bm25_index, build_tfidf_index, normalize_rows


def test_tfidf_and_bm25_index_build():
    texts = ["T1 PowerShell execution", "T2 file transfer"]
    vectorizer, matrix, _ = build_tfidf_index(
        texts,
        {
            "lowercase": True,
            "ngram_range": [1, 2],
            "min_df": 1,
            "sublinear_tf": True,
            "max_features": 50000,
        },
    )
    assert matrix.shape[0] == 2
    assert "powershell" in vectorizer.vocabulary_
    bm25, tokens, _ = build_bm25_index(texts)
    assert len(tokens) == 2
    assert len(bm25.get_scores(["powershell"])) == 2


def test_normalize_rows():
    matrix = normalize_rows(np.array([[3.0, 4.0], [0.0, 0.0]]))
    assert np.allclose(matrix[0], [0.6, 0.8])
    assert np.allclose(matrix[1], [0.0, 0.0])
