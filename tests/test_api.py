"""Tests for FastAPI endpoints, request validation, and status codes."""

from fastapi.testclient import TestClient


def test_service_info_endpoint(test_client: TestClient):
    """Test GET / returns 200 and valid service metadata."""
    response = test_client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "service" in data
    assert "version" in data
    assert data["docs_url"] == "/docs"


def test_health_endpoint(test_client: TestClient):
    """Test GET /health returns 200 and healthy status."""
    response = test_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["catalog_size"] == 5
    assert data["model_loaded"] is True


def test_genres_endpoint(test_client: TestClient):
    """Test GET /genres returns available genres list."""
    response = test_client.get("/genres")
    assert response.status_code == 200
    data = response.json()
    assert "genres" in data
    assert data["count"] > 0
    assert "Action" in data["genres"] or "Science Fiction" in data["genres"]


def test_search_movies_endpoint(test_client: TestClient):
    """Test GET /movies/search with valid query."""
    response = test_client.get("/movies/search?q=Gotham")
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "Gotham"
    assert data["total_found"] >= 1
    assert data["results"][0]["title"] == "Shadows of Gotham"


def test_search_movies_validation(test_client: TestClient):
    """Test GET /movies/search without required query param returns 422."""
    response = test_client.get("/movies/search")
    assert response.status_code == 422


def test_get_movie_details(test_client: TestClient):
    """Test GET /movies/{movie_id} with existing and non-existing IDs."""
    response = test_client.get("/movies/101")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == 101
    assert data["title"] == "Interstellar Odyssey"
    assert data["director"] == "Christopher Nolan"
    assert "Action" in data["genres"] or "Science Fiction" in data["genres"]

    # Not found
    not_found = test_client.get("/movies/999999")
    assert not_found.status_code == 404
    assert "not found" in not_found.json()["detail"].lower()


def test_get_recommendations_endpoint(test_client: TestClient):
    """Test GET /recommendations/{movie_id} with query parameters."""
    response = test_client.get("/recommendations/101?top_n=2")
    assert response.status_code == 200
    data = response.json()
    assert data["query_movie_id"] == 101
    assert data["count"] <= 2
    assert len(data["recommendations"]) <= 2

    # Verify query movie is not in recommendations
    rec_ids = [r["id"] for r in data["recommendations"]]
    assert 101 not in rec_ids

    # Unknown movie
    unknown_response = test_client.get("/recommendations/999999")
    assert unknown_response.status_code == 404


def test_get_recommendations_by_title_endpoint(test_client: TestClient):
    """Test GET /recommendations/by-title/{title}."""
    response = test_client.get("/recommendations/by-title/Gotham?top_n=2")
    assert response.status_code == 200
    data = response.json()
    assert data["query_movie_title"] == "Shadows of Gotham"
    assert len(data["recommendations"]) > 0


def test_recommendations_validation(test_client: TestClient):
    """Test query parameter validation (top_n limits)."""
    # top_n exceeds MAX_TOP_N (50)
    response = test_client.get("/recommendations/101?top_n=999")
    assert response.status_code == 422

    # top_n < 1
    neg_response = test_client.get("/recommendations/101?top_n=0")
    assert neg_response.status_code == 422
