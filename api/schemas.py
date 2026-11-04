"""Pydantic schemas and response models for MovieInsight REST API."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ServiceInfoResponse(BaseModel):
    """Basic service and API documentation information."""

    service: str = Field(..., examples=["MovieInsight Recommendation API"])
    version: str = Field(..., examples=["1.0.0"])
    description: str = Field(..., examples=["Content-based movie recommendation REST API"])
    docs_url: str = Field(..., examples=["/docs"])
    health_url: str = Field(..., examples=["/health"])


class HealthResponse(BaseModel):
    """Health check response schema."""

    status: str = Field(..., examples=["healthy"])
    version: str = Field(..., examples=["1.0.0"])
    catalog_size: int = Field(..., examples=[4803])
    model_loaded: bool = Field(..., examples=[True])


class GenresResponse(BaseModel):
    """Available genres in the catalog."""

    count: int = Field(..., examples=[20])
    genres: List[str] = Field(..., examples=[["Action", "Adventure", "Comedy", "Drama"]])


class MovieSummary(BaseModel):
    """Concise movie summary returned in search results."""

    id: int = Field(..., examples=[19995])
    title: str = Field(..., examples=["Avatar"])
    release_date: str = Field(..., examples=["2009-12-10"])
    vote_average: float = Field(..., examples=[7.2])
    genres: List[str] = Field(..., examples=[["Action", "Adventure", "Fantasy"]])
    director: Optional[str] = Field(None, examples=["James Cameron"])


class MovieDetail(BaseModel):
    """Comprehensive movie details."""

    id: int = Field(..., examples=[19995])
    title: str = Field(..., examples=["Avatar"])
    overview: str = Field(..., examples=["In the 22nd century, a paraplegic Marine..."])
    genres: List[str] = Field(..., examples=[["Action", "Adventure", "Fantasy", "Science Fiction"]])
    director: Optional[str] = Field(None, examples=["James Cameron"])
    cast: List[str] = Field(..., examples=[["Sam Worthington", "Zoe Saldana", "Sigourney Weaver"]])
    release_date: str = Field(..., examples=["2009-12-10"])
    vote_average: float = Field(..., examples=[7.2])
    vote_count: int = Field(..., examples=[11800])
    popularity: float = Field(..., examples=[150.44])
    budget: int = Field(..., examples=[237000000])
    revenue: int = Field(..., examples=[2787965087])
    profit: int = Field(..., examples=[2550965087])
    roi: Optional[float] = Field(None, examples=[10.764])
    success: str = Field(..., examples=["Hit"])


class RecommendationExplanation(BaseModel):
    """Feature-level explainability breakdown for a recommendation."""

    summary: str = Field(..., examples=["Recommended based on shared genres (Action, Adventure) and director (James Cameron)."])
    shared_genres: List[str] = Field(..., examples=[["Action", "Adventure"]])
    shared_director: Optional[str] = Field(None, examples=["James Cameron"])
    shared_cast: List[str] = Field(..., examples=[["Sigourney Weaver"]])
    shared_keywords: List[str] = Field(..., examples=[["alien", "culture clash"]])


class RecommendationItem(MovieDetail):
    """Recommended movie item with similarity score and explanation."""

    similarity_score: float = Field(..., examples=[0.3421])
    explanation: RecommendationExplanation


class RecommendationResponse(BaseModel):
    """Response payload for recommendation queries."""

    query_movie_id: int = Field(..., examples=[19995])
    query_movie_title: str = Field(..., examples=["Avatar"])
    genre_filter_applied: Optional[str] = Field(None, examples=["Action"])
    count: int = Field(..., examples=[10])
    recommendations: List[RecommendationItem]


class SearchResponse(BaseModel):
    """Response payload for movie search queries."""

    query: str = Field(..., examples=["Inception"])
    total_found: int = Field(..., examples=[1])
    results: List[MovieSummary]


class ErrorResponse(BaseModel):
    """Standardized API error message."""

    detail: str = Field(..., examples=["Movie with ID 999999 not found in database."])
