"""Pytest fixtures and mock dataset generators for MovieInsight tests."""

import sys
from pathlib import Path
from typing import Generator
import pandas as pd
import pytest
from fastapi.testclient import TestClient

# Ensure src and api are in python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from api.main import app
from movieinsight.preprocessing import preprocess_data
from movieinsight.recommender import MovieRecommender


@pytest.fixture
def sample_raw_movies_df() -> pd.DataFrame:
    """Synthetic raw movies DataFrame resembling TMDB schema."""
    return pd.DataFrame(
        [
            {
                "id": 101,
                "title": "Interstellar Odyssey",
                "genres": '[{"id": 878, "name": "Science Fiction"}, {"id": 12, "name": "Adventure"}]',
                "keywords": '[{"id": 1, "name": "space"}, {"id": 2, "name": "wormhole"}]',
                "overview": "A team of explorers travel through a wormhole in space to ensure humanity's survival.",
                "budget": 165000000,
                "revenue": 677471339,
                "popularity": 95.5,
                "vote_average": 8.6,
                "vote_count": 12000,
                "release_date": "2014-11-05",
            },
            {
                "id": 102,
                "title": "Cosmic Voyager",
                "genres": '[{"id": 878, "name": "Science Fiction"}, {"id": 12, "name": "Adventure"}]',
                "keywords": '[{"id": 1, "name": "space"}, {"id": 3, "name": "astronaut"}]',
                "overview": "An astronaut navigates deep cosmic hazards to reach a new galaxy.",
                "budget": 100000000,
                "revenue": 350000000,
                "popularity": 80.2,
                "vote_average": 7.9,
                "vote_count": 8500,
                "release_date": "2016-09-12",
            },
            {
                "id": 103,
                "title": "Shadows of Gotham",
                "genres": '[{"id": 28, "name": "Action"}, {"id": 80, "name": "Crime"}, {"id": 18, "name": "Drama"}]',
                "keywords": '[{"id": 4, "name": "vigilante"}, {"id": 5, "name": "dark knight"}]',
                "overview": "A masked vigilante protects the city from organized crime syndicates.",
                "budget": 185000000,
                "revenue": 1004558444,
                "popularity": 110.4,
                "vote_average": 9.0,
                "vote_count": 19000,
                "release_date": "2008-07-18",
            },
            {
                "id": 104,
                "title": "The Dark Detective",
                "genres": '[{"id": 28, "name": "Action"}, {"id": 80, "name": "Crime"}]',
                "keywords": '[{"id": 4, "name": "vigilante"}, {"id": 6, "name": "detective"}]',
                "overview": "A detective battles corruption and underground criminals in a decaying city.",
                "budget": 150000000,
                "revenue": 550000000,
                "popularity": 75.0,
                "vote_average": 7.5,
                "vote_count": 6000,
                "release_date": "2011-04-20",
            },
            {
                "id": 105,
                "title": "Love in Paris",
                "genres": '[{"id": 10749, "name": "Romance"}, {"id": 35, "name": "Comedy"}]',
                "keywords": '[{"id": 7, "name": "paris"}, {"id": 8, "name": "wedding"}]',
                "overview": "Two strangers fall in love amidst the romantic streets and cafes of Paris.",
                "budget": 20000000,
                "revenue": 15000000,
                "popularity": 45.0,
                "vote_average": 6.8,
                "vote_count": 1200,
                "release_date": "2019-02-14",
            },
        ]
    )


@pytest.fixture
def sample_raw_credits_df() -> pd.DataFrame:
    """Synthetic raw credits DataFrame matching sample movies."""
    return pd.DataFrame(
        [
            {
                "movie_id": 101,
                "title": "Interstellar Odyssey",
                "cast": '[{"name": "Matthew McConaughey"}, {"name": "Anne Hathaway"}, {"name": "Jessica Chastain"}]',
                "crew": '[{"job": "Director", "name": "Christopher Nolan"}]',
            },
            {
                "movie_id": 102,
                "title": "Cosmic Voyager",
                "cast": '[{"name": "Matthew McConaughey"}, {"name": "Brad Pitt"}, {"name": "Sandra Bullock"}]',
                "crew": '[{"job": "Director", "name": "Alfonso Cuarón"}]',
            },
            {
                "movie_id": 103,
                "title": "Shadows of Gotham",
                "cast": '[{"name": "Christian Bale"}, {"name": "Heath Ledger"}, {"name": "Michael Caine"}]',
                "crew": '[{"job": "Director", "name": "Christopher Nolan"}]',
            },
            {
                "movie_id": 104,
                "title": "The Dark Detective",
                "cast": '[{"name": "Christian Bale"}, {"name": "Gary Oldman"}]',
                "crew": '[{"job": "Director", "name": "Matt Reeves"}]',
            },
            {
                "movie_id": 105,
                "title": "Love in Paris",
                "cast": '[{"name": "Marion Cotillard"}, {"name": "Owen Wilson"}]',
                "crew": '[{"job": "Director", "name": "Woody Allen"}]',
            },
        ]
    )


@pytest.fixture
def sample_merged_df(sample_raw_movies_df, sample_raw_credits_df) -> pd.DataFrame:
    return sample_raw_movies_df.merge(
        sample_raw_credits_df, left_on="id", right_on="movie_id", suffixes=("", "_credits")
    )


@pytest.fixture
def sample_processed_df(sample_merged_df) -> pd.DataFrame:
    """Preprocessed DataFrame ready for recommender."""
    return preprocess_data(sample_merged_df)


@pytest.fixture
def sample_recommender(sample_processed_df) -> MovieRecommender:
    """Fitted MovieRecommender instance."""
    recommender = MovieRecommender()
    recommender.fit(sample_processed_df)
    return recommender


@pytest.fixture
def test_client(sample_recommender) -> Generator[TestClient, None, None]:
    """FastAPI TestClient with injected sample recommender."""
    app.state.recommender = sample_recommender
    with TestClient(app) as client:
        yield client
