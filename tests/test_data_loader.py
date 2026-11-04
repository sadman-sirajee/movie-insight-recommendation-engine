"""Tests for data loading, validation, and file error handling."""

from pathlib import Path
import pandas as pd
import pytest

from movieinsight.data_loader import DataLoadingError, load_raw_data, validate_raw_files


def test_validate_raw_files_missing(tmp_path: Path):
    """Test that missing files raise DataLoadingError."""
    fake_movies = tmp_path / "non_existent_movies.csv"
    fake_credits = tmp_path / "non_existent_credits.csv"

    with pytest.raises(DataLoadingError, match="Missing movies dataset"):
        validate_raw_files(fake_movies, fake_credits)


def test_validate_raw_files_empty(tmp_path: Path):
    """Test that zero-byte empty files raise DataLoadingError."""
    empty_movies = tmp_path / "empty_movies.csv"
    empty_credits = tmp_path / "empty_credits.csv"
    empty_movies.touch()
    empty_credits.touch()

    with pytest.raises(DataLoadingError, match="is empty"):
        validate_raw_files(empty_movies, empty_credits)


def test_load_raw_data_valid(tmp_path: Path, sample_raw_movies_df, sample_raw_credits_df):
    """Test that valid CSVs are loaded and merged correctly."""
    movies_path = tmp_path / "movies.csv"
    credits_path = tmp_path / "movies_credits.csv"

    sample_raw_movies_df.to_csv(movies_path, index=False)
    sample_raw_credits_df.to_csv(credits_path, index=False)

    df = load_raw_data(movies_path=movies_path, credits_path=credits_path)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 5
    assert "id" in df.columns
    assert "title" in df.columns
    assert "cast" in df.columns
    assert "crew" in df.columns


def test_load_raw_data_missing_column(tmp_path: Path):
    """Test that dataset missing essential columns raises DataLoadingError."""
    bad_movies = tmp_path / "bad_movies.csv"
    bad_credits = tmp_path / "bad_credits.csv"

    pd.DataFrame({"id": [1], "some_col": ["val"]}).to_csv(bad_movies, index=False)
    pd.DataFrame({"movie_id": [1], "title": ["T"], "cast": ["[]"], "crew": ["[]"]}).to_csv(bad_credits, index=False)

    with pytest.raises(DataLoadingError, match="missing required columns"):
        load_raw_data(movies_path=bad_movies, credits_path=bad_credits)


def test_load_raw_data_deduplication(tmp_path: Path, sample_raw_movies_df, sample_raw_credits_df):
    """Test that duplicate IDs in movies dataset are cleaned."""
    dup_movies = pd.concat([sample_raw_movies_df, sample_raw_movies_df.iloc[[0]]], ignore_index=True)
    movies_path = tmp_path / "dup_movies.csv"
    credits_path = tmp_path / "credits.csv"

    dup_movies.to_csv(movies_path, index=False)
    sample_raw_credits_df.to_csv(credits_path, index=False)

    df = load_raw_data(movies_path=movies_path, credits_path=credits_path)
    assert len(df) == 5
    assert df["id"].nunique() == 5
