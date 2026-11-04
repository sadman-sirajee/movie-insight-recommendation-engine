"""Evaluation script to run diagnostic benchmarks on MovieInsight recommender."""

import json
import logging
import sys
from pathlib import Path

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from movieinsight.data_loader import load_raw_data
from movieinsight.evaluation import run_benchmark_evaluation
from movieinsight.preprocessing import preprocess_data
from movieinsight.recommender import MovieRecommender

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluator")


def main():
    logger.info("Starting MovieInsight evaluation pipeline...")
    try:
        raw_df = load_raw_data()
        processed_df = preprocess_data(raw_df)
    except Exception as exc:
        logger.error("Failed to load or preprocess dataset: %s", exc)
        sys.exit(1)

    recommender = MovieRecommender()
    recommender.fit(processed_df)

    logger.info("Running diagnostic benchmark...")
    results = run_benchmark_evaluation(recommender, sample_size=50, k=10)

    print("\n=======================================================")
    print("           MOVIEINSIGHT EVALUATION REPORT             ")
    print("=======================================================")
    print(f"Total Movies in Catalog:       {results['total_catalog_size']}")
    print(f"Sample Query Size:             {results['sample_size']}")
    print(f"Rank Cutoff (k):               {results['k']}")
    print(f"Mean Genre Precision@{results['k']}:     {results['mean_genre_precision_at_k']:.4f}")
    print(f"Mean Genre Jaccard@{results['k']}:       {results['mean_genre_jaccard_at_k']:.4f}")
    print(f"Catalog Coverage:              {results['catalog_coverage_pct']:.2f}% ({results['distinct_items_recommended']} distinct movies)")
    print("\n--- Known Franchise Sequel Retrieval Benchmark ---")
    franchise = results["franchise_retrieval"]
    print(f"Franchise Hit Rate@{results['k']}:       {franchise['hit_rate_at_k']:.4f}")
    print(f"Franchise MRR@{results['k']}:            {franchise['mrr_at_k']:.4f} (Mean Reciprocal Rank)")
    print(f"Evaluated Test Pairs:          {franchise['evaluated_pairs']}")
    print("\n--- Diagnostic Methodology & Assumptions ---")
    for lim in results["limitations_and_assumptions"]:
        print(f"- {lim}")
    print("=======================================================\n")


if __name__ == "__main__":
    main()
