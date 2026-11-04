"""Tests for data preprocessing, JSON extraction, and financial calculations."""

import numpy as np
import pandas as pd
import pytest

from movieinsight.preprocessing import (
    calculate_financials,
    create_metadata_soup,
    extract_director,
    preprocess_data,
    safe_parse_json_list,
)


def test_safe_parse_json_list_variations():
    """Test JSON parsing with valid, empty, malformed, and non-string inputs."""
    # Valid Python literal
    lit = '[{"id": 1, "name": "Action"}, {"id": 2, "name": "Adventure"}]'
    assert safe_parse_json_list(lit) == ["Action", "Adventure"]

    # Valid JSON string
    js = '[{"name": "Drama"}, {"name": "Comedy"}]'
    assert safe_parse_json_list(js) == ["Drama", "Comedy"]

    # Max items truncation
    assert safe_parse_json_list(lit, max_items=1) == ["Action"]

    # Empty string and None
    assert safe_parse_json_list("") == []
    assert safe_parse_json_list(None) == []
    assert safe_parse_json_list(np.nan) == []

    # Malformed string
    assert safe_parse_json_list("not a valid json [[{") == []

    # Already parsed list
    assert safe_parse_json_list(["Sci-Fi", "Mystery"]) == ["Sci-Fi", "Mystery"]


def test_extract_director():
    """Test director extraction from crew metadata."""
    crew_str = '[{"job": "Producer", "name": "Emma Thomas"}, {"job": "Director", "name": "Christopher Nolan"}]'
    assert extract_director(crew_str) == "Christopher Nolan"

    # No director present
    crew_no_dir = '[{"job": "Producer", "name": "Emma Thomas"}]'
    assert extract_director(crew_no_dir) is None

    # Empty and None
    assert extract_director("") is None
    assert extract_director(None) is None
    assert extract_director(np.nan) is None


def test_calculate_financials():
    """Test profit, ROI, and success categorization."""
    df = pd.DataFrame(
        {
            "budget": [100, 100, 100, 0, None],
            "revenue": [300, 150, 50, 500, 100],
        }
    )
    result = calculate_financials(df)

    # Row 0: budget 100, rev 300 -> profit 200, roi 2.0 -> "Hit"
    assert result["profit"].iloc[0] == 200
    assert result["roi"].iloc[0] == 2.0
    assert result["success"].iloc[0] == "Hit"

    # Row 1: budget 100, rev 150 -> profit 50, roi 0.5 -> "Average"
    assert result["profit"].iloc[1] == 50
    assert result["roi"].iloc[1] == 0.5
    assert result["success"].iloc[1] == "Average"

    # Row 2: budget 100, rev 50 -> profit -50, roi -0.5 -> "Flop"
    assert result["profit"].iloc[2] == -50
    assert result["roi"].iloc[2] == -0.5
    assert result["success"].iloc[2] == "Flop"

    # Row 3: budget 0 -> roi NaN -> "Unknown"
    assert pd.isna(result["roi"].iloc[3])
    assert result["success"].iloc[3] == "Unknown"


def test_create_metadata_soup():
    """Test metadata soup token generation and multi-word joining."""
    soup = create_metadata_soup(
        overview="Space explorers embark on a journey.",
        genres=["Science Fiction", "Adventure"],
        keywords=["wormhole", "time travel"],
        cast=["Matthew McConaughey", "Anne Hathaway"],
        director="Christopher Nolan",
    )

    # Multi-word tokens should be underscore joined
    assert "science_fiction" in soup
    assert "christopher_nolan" in soup
    assert "matthew_mcconaughey" in soup
    assert "time_travel" in soup
    # Director should be repeated for weighting
    assert soup.count("christopher_nolan") == 2


def test_preprocess_data_pipeline(sample_merged_df):
    """Test end-to-end preprocessing pipeline on sample dataset."""
    df = preprocess_data(sample_merged_df)

    assert len(df) == 5
    for col in ["id", "title", "soup", "genres", "director", "cast", "profit", "roi", "success"]:
        assert col in df.columns
    assert all(isinstance(s, str) and len(s) > 0 for s in df["soup"])
