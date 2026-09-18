import numpy as np

from src.attention import attention_token_rows, crop_attention_matrix


def test_attention_token_rows_masks_special_tokens():
    rows = attention_token_rows(["[CLS]", "power", "##shell"], [101, 200, 201], np.array([0.5, 0.3, 0.2]), {101})
    assert rows[0]["is_special"] is True
    assert rows[1]["token"] == "power"
    assert rows[2]["cls_attention"] == 0.2


def test_crop_attention_matrix():
    matrix, tokens = crop_attention_matrix(np.ones((4, 4)), ["a", "b", "c", "d"], 2)
    assert matrix.shape == (2, 2)
    assert tokens == ["a", "b"]
