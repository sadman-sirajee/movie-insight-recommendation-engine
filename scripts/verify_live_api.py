"""Live end-to-end FastAPI endpoint verification probe using ASGI TestClient with full dataset."""

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("api_verifier")

from fastapi.testclient import TestClient
from api.main import app


def main():
    logger.info("Initializing TestClient and loading recommendation engine with full dataset...")
    test_results = []

    def check(name, success, detail=""):
        test_results.append((name, success, detail))
        mark = "PASS" if success else "FAIL"
        print(f"  [{mark}] {name}: {detail}")

    with TestClient(app) as client:
        print("\n================ LIVE API ENDPOINT VERIFICATION ================")

        # 1. Root /
        r = client.get("/")
        b = r.json()
        check("GET / (Service Info)", r.status_code == 200 and "docs_url" in b, f"status={r.status_code}, service='{b.get('service')}'")

        # 2. Health
        r = client.get("/health")
        b = r.json()
        check("GET /health (Health & Metrics)", r.status_code == 200 and b.get("model_loaded") is True, f"status={r.status_code}, catalog_size={b.get('catalog_size')}")

        # 3. Genres
        r = client.get("/genres")
        b = r.json()
        check("GET /genres (Available Genres)", r.status_code == 200 and b.get("count", 0) > 10, f"status={r.status_code}, total_genres={b.get('count')}")

        # 4. Search
        r = client.get("/movies/search?q=Dark%20Knight")
        b = r.json()
        has_match = r.status_code == 200 and b.get("total_found", 0) >= 1
        sample_id = b["results"][0]["id"] if has_match else None
        check("GET /movies/search?q=Dark Knight", has_match, f"status={r.status_code}, total_found={b.get('total_found')}, top='{b['results'][0]['title'] if has_match else 'None'}'")

        # 5. Movie Details
        if sample_id:
            r = client.get(f"/movies/{sample_id}")
            b = r.json()
            check(f"GET /movies/{sample_id} (Details)", r.status_code == 200 and "title" in b and "director" in b, f"title='{b.get('title')}', director='{b.get('director')}', rating={b.get('vote_average')}")

        # 6. Recommendations by ID
        if sample_id:
            r = client.get(f"/recommendations/{sample_id}?top_n=3")
            b = r.json()
            recs = b.get("recommendations", [])
            valid_recs = r.status_code == 200 and len(recs) == 3 and "explanation" in recs[0]
            top_rec_title = recs[0]["title"] if recs else ""
            top_rec_score = recs[0].get("similarity_score") if recs else 0
            check(f"GET /recommendations/{sample_id}?top_n=3", valid_recs, f"status={r.status_code}, returned={len(recs)}, top_match='{top_rec_title}' (score={top_rec_score})")
            if recs:
                print(f"      Explainability Reason: {recs[0]['explanation']['summary']}")

        # 7. Recommendations with Genre Filter
        if sample_id:
            r = client.get(f"/recommendations/{sample_id}?top_n=2&genre=Action")
            b = r.json()
            recs = b.get("recommendations", [])
            all_action = all("Action" in rec.get("genres", []) for rec in recs)
            check(f"GET /recommendations/{sample_id}?genre=Action", r.status_code == 200 and all_action and len(recs) == 2, f"status={r.status_code}, returned {len(recs)} Action recommendations")

        # 8. Recommendations by Title
        r = client.get("/recommendations/by-title/Avatar?top_n=2")
        b = r.json()
        recs = b.get("recommendations", [])
        check("GET /recommendations/by-title/Avatar", r.status_code == 200 and len(recs) == 2, f"status={r.status_code}, query_title='{b.get('query_movie_title')}'")

        # 9. Error 404 for unknown movie
        r = client.get("/movies/9999999")
        b = r.json()
        check("GET /movies/9999999 (404 Error Handling)", r.status_code == 404, f"status={r.status_code}, detail='{b.get('detail')}'")

        # 10. Error 422 for invalid parameter (top_n > 50)
        r = client.get(f"/recommendations/{sample_id or 101}?top_n=999")
        check("Query validation top_n=999 (422 Handling)", r.status_code == 422, f"status={r.status_code}")

        # 11. Error 422 for missing required query
        r = client.get("/movies/search")
        check("Query validation missing search q (422 Handling)", r.status_code == 422, f"status={r.status_code}")

        print("=================================================================\n")

        all_passed = all(success for _, success, _ in test_results)
        if all_passed:
            logger.info("ALL %d LIVE ENDPOINTS VERIFIED AND RESPONDING CORRECTLY!", len(test_results))
        else:
            logger.error("Some live endpoint checks failed.")
            sys.exit(1)


if __name__ == "__main__":
    main()
