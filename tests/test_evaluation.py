"""Tests for evaluation metrics and benchmark suite."""

from movieinsight.evaluation import (
    compute_catalog_coverage,
    compute_genre_jaccard_at_k,
    compute_genre_precision_at_k,
    evaluate_known_item_retrieval,
    run_benchmark_evaluation,
)


def test_compute_genre_precision_at_k():
    """Test Precision@K on synthetic recommendation items."""
    seed = ["Action", "Sci-Fi"]
    recs = [
        {"genres": ["Action", "Thriller"]},  # Match (1)
        {"genres": ["Sci-Fi"]},              # Match (2)
        {"genres": ["Romance", "Comedy"]},   # No match (0)
        {"genres": ["Action"]},              # Match (3)
    ]

    p_at_4 = compute_genre_precision_at_k(seed, recs, k=4)
    assert p_at_4 == 0.75

    p_at_2 = compute_genre_precision_at_k(seed, recs, k=2)
    assert p_at_2 == 1.0


def test_compute_genre_jaccard_at_k():
    """Test average Jaccard similarity across recommendations."""
    seed = ["Action", "Adventure"]
    recs = [
        {"genres": ["Action", "Adventure"]},  # 2/2 = 1.0
        {"genres": ["Action"]},               # 1/2 = 0.5
    ]
    j = compute_genre_jaccard_at_k(seed, recs, k=2)
    assert j == 0.75


def test_compute_catalog_coverage():
    """Test coverage computation."""
    recommended_ids = {1, 2, 3, 4, 5}
    coverage = compute_catalog_coverage(recommended_ids, total_catalog_size=10)
    assert coverage == 50.0

    assert compute_catalog_coverage(set(), total_catalog_size=10) == 0.0
    assert compute_catalog_coverage(recommended_ids, total_catalog_size=0) == 0.0


def test_evaluate_known_item_retrieval(sample_recommender):
    """Test known-item retrieval evaluation on sample recommender."""
    test_pairs = [
        {
            "query_id": 101,  # Interstellar Odyssey
            "expected_ids": [102],  # Cosmic Voyager
        }
    ]
    results = evaluate_known_item_retrieval(sample_recommender, test_pairs, k=5)
    assert results["evaluated_pairs"] == 1
    assert results["hit_rate_at_k"] == 1.0
    assert results["mrr_at_k"] > 0.0


def test_run_benchmark_evaluation(sample_recommender):
    """Test full benchmark evaluation run on sample dataset."""
    benchmark = run_benchmark_evaluation(sample_recommender, sample_size=3, k=2)

    assert "mean_genre_precision_at_k" in benchmark
    assert "catalog_coverage_pct" in benchmark
    assert "limitations_and_assumptions" in benchmark
    assert len(benchmark["limitations_and_assumptions"]) >= 3
