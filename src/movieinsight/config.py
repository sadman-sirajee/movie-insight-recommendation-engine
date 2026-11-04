"""Configuration settings and environment variable definitions for MovieInsight."""

import os
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR", PROJECT_ROOT / "data"))
RAW_MOVIES_PATH = DATA_DIR / "movies.csv"
RAW_CREDITS_PATH = DATA_DIR / "movies_credits.csv"
ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", PROJECT_ROOT / "data" / "processed"))

# Model Configuration
DEFAULT_TOP_N = int(os.getenv("DEFAULT_TOP_N", "10"))
MAX_TOP_N = int(os.getenv("MAX_TOP_N", "50"))
TFIDF_MAX_FEATURES = int(os.getenv("TFIDF_MAX_FEATURES", "10000"))
TFIDF_STOP_WORDS = "english"
TFIDF_NGRAM_RANGE = (1, 2)

# API Server Configuration
API_HOST = os.getenv("HOST", "0.0.0.0")
API_PORT = int(os.getenv("PORT", "8000"))
API_TITLE = "MovieInsight Recommendation API"
API_VERSION = "1.0.0"
API_DESCRIPTION = (
    "Production-ready Movie Recommendation and Data Science REST API powered by "
    "TF-IDF, Cosine Similarity, and Feature Engineering on the TMDB dataset."
)
