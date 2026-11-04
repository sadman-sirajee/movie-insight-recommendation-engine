"""Pipeline runner to process raw data and test recommendation generation."""

import logging
import sys
from pathlib import Path

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from movieinsight.config import ARTIFACTS_DIR
from movieinsight.data_loader import load_raw_data
from movieinsight.preprocessing import preprocess_data
from movieinsight.recommender import MovieRecommender

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("pipeline")


def main():
    logger.info("Running MovieInsight Data Pipeline...")
    raw_df = load_raw_data()
    logger.info("Raw dataset loaded: %d rows", len(raw_df))

    processed_df = preprocess_data(raw_df)
    logger.info("Dataset preprocessed: %d rows", len(processed_df))

    # Save processed dataset to data/processed/processed_movies.csv
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = ARTIFACTS_DIR / "processed_movies.csv"
    processed_df.to_csv(out_csv, index=False)
    logger.info("Saved processed dataset to %s", out_csv)

    # Initialize and fit recommender
    recommender = MovieRecommender()
    recommender.fit(processed_df)

    # Test sample search and recommendation
    test_title = "Inception"
    logger.info("Testing recommendation for '%s'...", test_title)
    recs = recommender.recommend_by_title(test_title, top_n=5)

    print(f"\nTop 5 Recommendations for '{test_title}':")
    for idx, r in enumerate(recs, start=1):
        print(f" {idx}. {r['title']} (Score: {r['similarity_score']})")
        print(f"    Genres: {', '.join(r['genres'])}")
        print(f"    Director: {r['director']}")
        print(f"    Why: {r['explanation']['summary']}\n")

    logger.info("Pipeline executed successfully!")


if __name__ == "__main__":
    main()
