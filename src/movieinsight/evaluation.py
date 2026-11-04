"""Evaluation metrics and offline diagnostic benchmarks for MovieInsight."""

import logging
from typing import Any, Dict, List, Optional, Set
import numpy as np

from movieinsight.recommender import MovieRecommender

logger = logging.getLogger(__name__)


def compute_genre_precision_at_k(
    seed_genres: List[str],
    recommendations: List[Dict[str, Any]],
    k: int = 10,
) -> float:
    """Compute Precision@K based on genre consistency.

    A recommended movie is considered relevant under this heuristic if it shares
    at least one genre with the seed movie.

    Args:
        seed_genres: Genres of the query movie.
        recommendations: Recommended movie items.
        k: Cutoff rank.

    Returns:
        float: Precision@K in [0.0, 1.0].
    """
    if not seed_genres or not recommendations or k <= 0:
        return 0.0

    seed_set = set(g.lower() for g in seed_genres)
    top_k = recommendations[:k]
    relevant_count = 0

    for rec in top_k:
        rec_genres = set(g.lower() for g in rec.get("genres", []))
        if seed_set.intersection(rec_genres):
            relevant_count += 1

    return relevant_count / len(top_k)


def compute_genre_jaccard_at_k(
    seed_genres: List[str],
    recommendations: List[Dict[str, Any]],
    k: int = 10,
) -> float:
    """Compute average Jaccard genre similarity between seed and top-K recommendations.

    Jaccard = |G_seed ∩ G_rec| / |G_seed ∪ G_rec|

    Args:
        seed_genres: Genres of the query movie.
        recommendations: Recommended movie items.
        k: Cutoff rank.

    Returns:
        float: Mean Jaccard similarity across top-K recommendations.
    """
    if not seed_genres or not recommendations or k <= 0:
        return 0.0

    seed_set = set(g.lower() for g in seed_genres)
    top_k = recommendations[:k]
    jaccard_scores: List[float] = []

    for rec in top_k:
        rec_genres = set(g.lower() for g in rec.get("genres", []))
        union_len = len(seed_set.union(rec_genres))
        if union_len == 0:
            jaccard_scores.append(0.0)
        else:
            jaccard_scores.append(len(seed_set.intersection(rec_genres)) / union_len)

    return float(np.mean(jaccard_scores)) if jaccard_scores else 0.0


def compute_catalog_coverage(
    all_recommended_ids: Set[int],
    total_catalog_size: int,
) -> float:
    """Compute Catalog Coverage as percentage of distinct catalog items recommended.

    Args:
        all_recommended_ids: Set of unique movie IDs recommended across seed queries.
        total_catalog_size: Total number of movies in the catalog.

    Returns:
        float: Coverage percentage in [0.0, 100.0].
    """
    if total_catalog_size <= 0:
        return 0.0
    return (len(all_recommended_ids) / total_catalog_size) * 100.0


def evaluate_known_item_retrieval(
    recommender: MovieRecommender,
    test_pairs: List[Dict[str, Any]],
    k: int = 10,
) -> Dict[str, float]:
    """Evaluate Known-Item / Franchise Retrieval (Hit Rate@K and Mean Reciprocal Rank - MRR).

    Tests whether related sequels or known companion films appear in top-K recommendations.

    Args:
        recommender: Fitted MovieRecommender.
        test_pairs: List of dicts with 'query_id' (or 'query_title') and 'expected_ids' (or 'expected_titles').
        k: Rank cutoff.

    Returns:
        Dict[str, float]: Hit Rate@K and MRR@K.
    """
    hits = 0
    reciprocal_ranks: List[float] = []

    for pair in test_pairs:
        query_id = pair.get("query_id")
        if query_id is None and "query_title" in pair:
            m = recommender.search_movies(pair["query_title"], limit=1)
            if m:
                query_id = m[0]["id"]

        if query_id is None:
            continue

        expected_ids = set(pair.get("expected_ids", []))
        if "expected_titles" in pair:
            for t in pair["expected_titles"]:
                found = recommender.search_movies(t, limit=1)
                if found:
                    expected_ids.add(found[0]["id"])

        try:
            recs = recommender.recommend_by_id(query_id, top_n=k)
        except Exception:
            continue

        rank_found = 0
        for rank, rec in enumerate(recs[:k], start=1):
            if rec["id"] in expected_ids:
                rank_found = rank
                break

        if rank_found > 0:
            hits += 1
            reciprocal_ranks.append(1.0 / rank_found)
        else:
            reciprocal_ranks.append(0.0)

    total = len(test_pairs)
    hit_rate = (hits / total) if total > 0 else 0.0
    mrr = float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0

    return {
        "hit_rate_at_k": round(hit_rate, 4),
        "mrr_at_k": round(mrr, 4),
        "evaluated_pairs": total,
    }


def run_benchmark_evaluation(
    recommender: MovieRecommender,
    sample_size: int = 50,
    k: int = 10,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Run an end-to-end reproducible diagnostic evaluation benchmark.

    Evaluates:
    - Average Genre Precision@K
    - Average Genre Jaccard Similarity@K
    - Catalog Coverage across sampled seeds
    - Franchise Sequel Retrieval Benchmark (The Dark Knight, The Godfather, Toy Story, etc.)

    Args:
        recommender: Fitted MovieRecommender instance.
        sample_size: Number of random seed movies to evaluate.
        k: Recommendation rank cutoff.
        random_state: Random seed for reproducibility.

    Returns:
        Dict[str, Any]: Benchmark results with methodology notes and limitations.
    """
    if not recommender.is_fitted():
        raise RuntimeError("Recommender must be fitted before running evaluation.")

    df = recommender.movies_df
    catalog_size = len(df)
    sample_n = min(sample_size, catalog_size)

    sample_df = df.sample(n=sample_n, random_state=random_state)

    precision_list = []
    jaccard_list = []
    recommended_id_set: Set[int] = set()

    for _, row in sample_df.iterrows():
        movie_id = int(row["id"])
        genres = row.get("genres", [])
        recs = recommender.recommend_by_id(movie_id, top_n=k)

        p = compute_genre_precision_at_k(genres, recs, k=k)
        j = compute_genre_jaccard_at_k(genres, recs, k=k)

        precision_list.append(p)
        jaccard_list.append(j)

        for r in recs:
            recommended_id_set.add(r["id"])

    coverage = compute_catalog_coverage(recommended_id_set, catalog_size)

    # Known franchise benchmark pairs available in TMDB 5000
    known_franchise_tests = [
        {
            "query_title": "The Dark Knight",
            "expected_titles": ["Batman Begins", "The Dark Knight Rises"],
        },
        {
            "query_title": "The Godfather",
            "expected_titles": ["The Godfather: Part II", "The Godfather: Part III"],
        },
        {
            "query_title": "Toy Story",
            "expected_titles": ["Toy Story 2", "Toy Story 3"],
        },
        {
            "query_title": "Iron Man",
            "expected_titles": ["Iron Man 2", "Iron Man 3", "The Avengers"],
        },
        {
            "query_title": "The Lord of the Rings: The Fellowship of the Ring",
            "expected_titles": [
                "The Lord of the Rings: The Two Towers",
                "The Lord of the Rings: The Return of the King",
            ],
        },
    ]

    retrieval_results = evaluate_known_item_retrieval(
        recommender,
        known_franchise_tests,
        k=k,
    )

    return {
        "k": k,
        "sample_size": sample_n,
        "total_catalog_size": catalog_size,
        "mean_genre_precision_at_k": round(float(np.mean(precision_list)), 4),
        "mean_genre_jaccard_at_k": round(float(np.mean(jaccard_list)), 4),
        "catalog_coverage_pct": round(coverage, 2),
        "distinct_items_recommended": len(recommended_id_set),
        "franchise_retrieval": retrieval_results,
        "limitations_and_assumptions": [
            "Dataset lacks explicit user-item interaction logs (e.g. user IDs, clicks, watch time).",
            "Genre precision serves as a proxy for domain relevance rather than personalized user preference.",
            "Catalog coverage reflects diversity across sample seeds, not global user traffic distribution.",
            "Future improvements could incorporate MovieLens ratings to evaluate NDCG@K and Recall@K on explicit ratings.",
        ],
    }
