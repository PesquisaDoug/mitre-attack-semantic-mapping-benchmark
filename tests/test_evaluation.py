from src.evaluation import metrics_for_ranking, parse_relevant_ids


def test_parse_relevant_ids():
    assert parse_relevant_ids('["T1", "T2"]') == {"T1", "T2"}


def test_metrics_for_multi_relevance_ranking():
    metrics = metrics_for_ranking(["T3", "T2", "T1", "T4"], {"T1", "T3"}, cutoff=10)
    assert metrics["recall@1"] == 0.5
    assert metrics["recall@5"] == 1.0
    assert metrics["hit@1"] == 1.0
    assert metrics["mrr@10"] == 1.0
    assert 0.0 <= metrics["ndcg@10"] <= 1.0
    assert metrics["map@10"] == (1 / 1 + 2 / 3) / 2
