"""FastAPI REST API application for MovieInsight movie recommendation service."""

import ast
from contextlib import asynccontextmanager
import logging
import sys
from pathlib import Path
from typing import Optional
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Add src to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from api.schemas import (
    ErrorResponse,
    GenresResponse,
    HealthResponse,
    MovieDetail,
    MovieSummary,
    RecommendationItem,
    RecommendationResponse,
    SearchResponse,
    ServiceInfoResponse,
)
from movieinsight.config import (
    API_DESCRIPTION,
    API_HOST,
    API_PORT,
    API_TITLE,
    API_VERSION,
    ARTIFACTS_DIR,
    DEFAULT_TOP_N,
    MAX_TOP_N,
)
from movieinsight.data_loader import load_raw_data
from movieinsight.preprocessing import preprocess_data
from movieinsight.recommender import MovieNotFoundError, MovieRecommender

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("movieinsight_api")


def initialize_recommender() -> MovieRecommender:
    """Initialize and fit the movie recommender on startup.

    If processed artifacts exist, loads them directly for fast warm startup.
    Otherwise, runs the full data loader and preprocessing pipeline.

    Returns:
        MovieRecommender: Fully fitted recommendation model.
    """
    processed_csv = ARTIFACTS_DIR / "processed_movies.csv"

    if processed_csv.exists() and processed_csv.stat().st_size > 0:
        logger.info("Found existing processed dataset at %s. Loading...", processed_csv)
        df = pd.read_csv(processed_csv)
        # Parse list-like string representations saved in CSV
        for col in ["genres", "keywords", "cast"]:
            if col in df.columns:
                df[col] = df[col].apply(
                    lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith("[") else []
                )
    else:
        logger.info("Processed dataset not found. Running end-to-end data pipeline...")
        raw_df = load_raw_data()
        df = preprocess_data(raw_df)
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(processed_csv, index=False)
        logger.info("Persisted preprocessed dataset to %s", processed_csv)

    recommender = MovieRecommender()
    recommender.fit(df)
    logger.info("MovieRecommender initialized successfully with %d movies", len(df))
    return recommender


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler to load and cache the recommendation model at startup."""
    logger.info("Starting up MovieInsight service...")
    if getattr(app.state, "recommender", None) is None:
        try:
            app.state.recommender = initialize_recommender()
        except Exception as exc:
            logger.error("Failed to initialize recommender during startup: %s", exc)
            app.state.recommender = None
    yield
    logger.info("Shutting down MovieInsight service...")


app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=API_DESCRIPTION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS for local dashboards and frontend clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_recommender(request: Request) -> MovieRecommender:
    """Dependency helper to retrieve the fitted recommender from app state."""
    recommender = getattr(request.app.state, "recommender", None)
    if recommender is None or not recommender.is_fitted():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Recommendation model is not currently loaded or ready.",
        )
    return recommender


@app.get(
    "/",
    response_model=ServiceInfoResponse,
    summary="Service Information",
    tags=["General"],
)
def get_service_info() -> ServiceInfoResponse:
    """Return basic information about the MovieInsight recommendation service."""
    return ServiceInfoResponse(
        service=API_TITLE,
        version=API_VERSION,
        description=API_DESCRIPTION,
        docs_url="/docs",
        health_url="/health",
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    tags=["General"],
)
def get_health(request: Request) -> HealthResponse:
    """Return service health status and catalog metrics."""
    recommender = getattr(request.app.state, "recommender", None)
    is_ready = recommender is not None and recommender.is_fitted()
    catalog_size = len(recommender.movies_df) if is_ready else 0

    return HealthResponse(
        status="healthy" if is_ready else "degraded",
        version=API_VERSION,
        catalog_size=catalog_size,
        model_loaded=is_ready,
    )


@app.get(
    "/genres",
    response_model=GenresResponse,
    summary="List Available Genres",
    tags=["Metadata"],
)
def get_genres(request: Request) -> GenresResponse:
    """Return all distinct movie genres present in the dataset."""
    recommender = get_recommender(request)
    genres = recommender.all_genres
    return GenresResponse(count=len(genres), genres=genres)


@app.get(
    "/movies/search",
    response_model=SearchResponse,
    summary="Search Movies by Title",
    tags=["Movies"],
)
def search_movies(
    request: Request,
    q: str = Query(..., min_length=1, max_length=100, description="Movie title search term"),
    limit: int = Query(10, ge=1, le=50, description="Maximum number of matches to return"),
) -> SearchResponse:
    """Search the catalog for movies matching the query string."""
    recommender = get_recommender(request)
    matches = recommender.search_movies(query=q, limit=limit)
    summaries = [
        MovieSummary(
            id=m["id"],
            title=m["title"],
            release_date=m["release_date"],
            vote_average=m["vote_average"],
            genres=m["genres"],
            director=m["director"],
        )
        for m in matches
    ]
    return SearchResponse(query=q, total_found=len(summaries), results=summaries)


@app.get(
    "/movies/{movie_id}",
    response_model=MovieDetail,
    summary="Get Movie Details by ID",
    responses={404: {"model": ErrorResponse}},
    tags=["Movies"],
)
def get_movie_details(
    movie_id: int,
    request: Request,
) -> MovieDetail:
    """Retrieve complete metadata for a movie by its TMDB movie ID."""
    recommender = get_recommender(request)
    movie = recommender.get_movie_by_id(movie_id)
    if movie is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Movie with ID {movie_id} was not found in the catalog.",
        )
    return MovieDetail(**movie)


@app.get(
    "/recommendations/{movie_id}",
    response_model=RecommendationResponse,
    summary="Get Recommendations by Movie ID",
    responses={404: {"model": ErrorResponse}},
    tags=["Recommendations"],
)
def get_recommendations(
    movie_id: int,
    request: Request,
    top_n: int = Query(DEFAULT_TOP_N, ge=1, le=MAX_TOP_N, description="Number of recommendations"),
    genre: Optional[str] = Query(None, description="Optional genre to filter recommendations"),
) -> RecommendationResponse:
    """Generate top-N content-based recommendations for a specified movie ID."""
    recommender = get_recommender(request)
    target_movie = recommender.get_movie_by_id(movie_id)
    if target_movie is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Movie with ID {movie_id} was not found in the catalog.",
        )

    try:
        recs = recommender.recommend_by_id(movie_id=movie_id, top_n=top_n, genre_filter=genre)
    except MovieNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    return RecommendationResponse(
        query_movie_id=movie_id,
        query_movie_title=target_movie["title"],
        genre_filter_applied=genre,
        count=len(recs),
        recommendations=[RecommendationItem(**r) for r in recs],
    )


@app.get(
    "/recommendations/by-title/{title}",
    response_model=RecommendationResponse,
    summary="Get Recommendations by Movie Title",
    responses={404: {"model": ErrorResponse}},
    tags=["Recommendations"],
)
def get_recommendations_by_title(
    title: str,
    request: Request,
    top_n: int = Query(DEFAULT_TOP_N, ge=1, le=MAX_TOP_N, description="Number of recommendations"),
    genre: Optional[str] = Query(None, description="Optional genre filter"),
) -> RecommendationResponse:
    """Generate top-N recommendations using movie title directly."""
    recommender = get_recommender(request)
    try:
        recs = recommender.recommend_by_title(title=title, top_n=top_n, genre_filter=genre)
    except MovieNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    matches = recommender.search_movies(title, limit=1)
    query_id = matches[0]["id"]
    query_title = matches[0]["title"]

    return RecommendationResponse(
        query_movie_id=query_id,
        query_movie_title=query_title,
        genre_filter_applied=genre,
        count=len(recs),
        recommendations=[RecommendationItem(**r) for r in recs],
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.main:app", host=API_HOST, port=API_PORT, reload=True)
