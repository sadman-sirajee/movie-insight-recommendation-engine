"""Tests for recommendation engine, search, and explainability."""

import pytest

from movieinsight.recommender import MovieNotFoundError, MovieRecommender


def test_recommender_unfitted():
    """Test that querying unfitted recommender raises RuntimeError."""
    rec = MovieRecommender()
    assert not rec.is_fitted()

    with pytest.raises(RuntimeError, match="must be fitted"):
        rec.get_movie_by_id(101)

    with pytest.raises(RuntimeError, match="must be fitted"):
        rec.search_movies("Interstellar")

    with pytest.raises(RuntimeError, match="must be fitted"):
        rec.recommend_by_id(101)


def test_recommender_fit(sample_recommender, sample_processed_df):
    """Test recommender fitting and indices initialization."""
    assert sample_recommender.is_fitted()
    assert len(sample_recommender.all_genres) > 0
    assert 101 in sample_recommender.id_to_idx
    assert "interstellar odyssey" in sample_recommender.title_to_ids


def test_get_movie_by_id(sample_recommender):
    """Test fetching movie by valid and invalid ID."""
    movie = sample_recommender.get_movie_by_id(101)
    assert movie is not None
    assert movie["id"] == 101
    assert movie["title"] == "Interstellar Odyssey"
    assert movie["director"] == "Christopher Nolan"

    missing = sample_recommender.get_movie_by_id(99999)
    assert missing is None


def test_search_movies(sample_recommender):
    """Test movie title search matching."""
    # Exact match
    results = sample_recommender.search_movies("Interstellar Odyssey")
    assert len(results) >= 1
    assert results[0]["id"] == 101

    # Substring match
    results = sample_recommender.search_movies("Gotham")
    assert len(results) == 1
    assert results[0]["title"] == "Shadows of Gotham"

    # Non-existent search
    assert sample_recommender.search_movies("NonExistentXYZ") == []

    # Empty search string
    assert sample_recommender.search_movies("") == []


def test_recommend_by_id(sample_recommender):
    """Test top-N recommendations for space movie 101."""
    recs = sample_recommender.recommend_by_id(101, top_n=2)
    assert len(recs) <= 2
    # Query movie itself must not be in recommendations
    assert all(r["id"] != 101 for r in recs)

    # Cosmic Voyager (ID 102) shares genre, director/cast, keywords with Interstellar
    top_rec = recs[0]
    assert "similarity_score" in top_rec
    assert top_rec["similarity_score"] > 0
    assert "explanation" in top_rec
    assert "summary" in top_rec["explanation"]
    assert len(top_rec["explanation"]["shared_genres"]) > 0


def test_recommend_by_id_unknown(sample_recommender):
    """Test recommendation for unknown movie ID raises MovieNotFoundError."""
    with pytest.raises(MovieNotFoundError, match="not found in database"):
        sample_recommender.recommend_by_id(99999)


def test_recommend_by_title(sample_recommender):
    """Test recommendation by title."""
    recs = sample_recommender.recommend_by_title("Gotham", top_n=2)
    assert len(recs) > 0
    assert all(r["title"] != "Shadows of Gotham" for r in recs)

    with pytest.raises(MovieNotFoundError, match="No movie matching title"):
        sample_recommender.recommend_by_title("Unknown Alien Film XYZ 1234")


def test_genre_filter(sample_recommender):
    """Test recommendations with genre filtering applied."""
    recs = sample_recommender.recommend_by_id(101, top_n=5, genre_filter="Action")
    for r in recs:
        assert any(g.lower() == "action" for g in r["genres"])
