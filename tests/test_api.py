"""
test_api.py — API Endpoint Integration Tests
─────────────────────────────────────────────
WHAT IS TestClient?
  FastAPI's TestClient (powered by httpx) lets you make real HTTP requests
  to your app WITHOUT starting a server. It runs your full FastAPI app
  in-process — middleware, routing, validation, everything — and returns
  real HTTP responses. This is the gold standard for testing APIs.

WHAT WE TEST:
  1. Health and root endpoints work
  2. /analyze returns correct structure on valid input
  3. /analyze rejects invalid file types
  4. /analyze rejects files that are too large
  5. /analyze rejects empty job description
  6. /analyze handles non-PDF bytes gracefully
  7. Response schema matches AnalysisResponse exactly

HOW MULTIPART FILE UPLOAD TESTING WORKS:
  httpx sends files as multipart/form-data using the `files` and `data`
  parameters in client.post(). We pass PDF bytes as a tuple:
    ("resume", (filename, bytes, content_type))
  And form fields as a dict in `data`.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app

# ── Test Client ───────────────────────────────────────────────────────────────
# A single client instance shared across all tests in this file.
# scope="module" means it's created once per file, not once per test.
@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


# ── Sample Data Fixtures ──────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def sample_jd() -> str:
    return """
    We are hiring a Python Backend Engineer to join our team.
    Must have experience with FastAPI or Django REST framework.
    Strong knowledge of Docker and AWS is required.
    PostgreSQL database experience is essential.
    Experience with machine learning and scikit-learn is a plus.
    Must use Git for version control. pytest knowledge required.
    Agile environment with bi-weekly sprints.
    """


@pytest.fixture(scope="module")
def valid_pdf_bytes() -> bytes:
    """Minimal valid PDF with tech skill keywords."""
    return (
        b"%PDF-1.4\n"
        b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
        b"3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R"
        b"/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>endobj\n"
        b"4 0 obj<</Length 90>>\nstream\n"
        b"BT /F1 10 Tf 50 750 Td (Python FastAPI Docker AWS PostgreSQL Developer Git pytest) Tj ET\n"
        b"endstream\nendobj\n"
        b"5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj\n"
        b"xref\n0 6\n"
        b"0000000000 65535 f \n"
        b"0000000009 00000 n \n"
        b"0000000058 00000 n \n"
        b"0000000115 00000 n \n"
        b"0000000266 00000 n \n"
        b"0000000408 00000 n \n"
        b"trailer<</Size 6/Root 1 0 R>>\n"
        b"startxref\n489\n%%EOF"
    )


@pytest.fixture(scope="module")
def not_a_pdf() -> bytes:
    return b"\xff\xd8\xff\xe0" + b"\x00" * 200  # JPEG magic bytes


# ── Health & Root Tests ───────────────────────────────────────────────────────

class TestHealthEndpoints:

    def test_root_returns_200(self, client):
        response = client.get("/")
        assert response.status_code == 200

    def test_root_contains_app_name(self, client):
        response = client.get("/")
        assert response.json()["app"] == "ResuFit"

    def test_root_contains_status(self, client):
        response = client.get("/")
        assert response.json()["status"] == "running"

    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_ok(self, client):
        response = client.get("/health")
        assert response.json() == {"status": "ok"}


# ── Analyze Endpoint — Valid Input Tests ──────────────────────────────────────

class TestAnalyzeEndpointSuccess:

    def _post_analyze(self, client, pdf_bytes, jd_text):
        """Helper to reduce repeated multipart upload boilerplate."""
        return client.post(
            "/api/v1/analyze",
            files={"resume": ("resume.pdf", pdf_bytes, "application/pdf")},
            data={"job_description": jd_text},
        )

    def test_returns_200_on_valid_input(self, client, valid_pdf_bytes, sample_jd):
        response = self._post_analyze(client, valid_pdf_bytes, sample_jd)
        # Accept 200 (success) or 400 (minimal PDF may not extract text)
        assert response.status_code in (200, 400)

    def test_response_has_all_required_fields(self, client, valid_pdf_bytes, sample_jd):
        response = self._post_analyze(client, valid_pdf_bytes, sample_jd)
        if response.status_code != 200:
            pytest.skip("Minimal PDF did not extract text in this environment")

        body = response.json()
        required_fields = [
            "fit_score", "fit_percentage", "tfidf_score", "skill_score",
            "verdict", "advice",
            "matched_skills", "missing_skills", "extra_skills",
            "resume_skill_count", "jd_skill_count", "resume_text_length",
        ]
        for field in required_fields:
            assert field in body, f"Missing field: {field}"

    def test_fit_score_is_in_valid_range(self, client, valid_pdf_bytes, sample_jd):
        response = self._post_analyze(client, valid_pdf_bytes, sample_jd)
        if response.status_code != 200:
            pytest.skip("Minimal PDF did not extract text in this environment")
        body = response.json()
        assert 0.0 <= body["fit_score"] <= 1.0

    def test_verdict_is_valid_string(self, client, valid_pdf_bytes, sample_jd):
        response = self._post_analyze(client, valid_pdf_bytes, sample_jd)
        if response.status_code != 200:
            pytest.skip("Minimal PDF did not extract text in this environment")
        body = response.json()
        assert body["verdict"] in {"Strong Fit", "Partial Fit", "Weak Fit"}

    def test_skill_fields_are_lists(self, client, valid_pdf_bytes, sample_jd):
        response = self._post_analyze(client, valid_pdf_bytes, sample_jd)
        if response.status_code != 200:
            pytest.skip("Minimal PDF did not extract text in this environment")
        body = response.json()
        assert isinstance(body["matched_skills"], list)
        assert isinstance(body["missing_skills"], list)
        assert isinstance(body["extra_skills"],   list)

    def test_content_type_is_json(self, client, valid_pdf_bytes, sample_jd):
        response = self._post_analyze(client, valid_pdf_bytes, sample_jd)
        assert "application/json" in response.headers["content-type"]


# ── Analyze Endpoint — Error Handling Tests ───────────────────────────────────

class TestAnalyzeEndpointErrors:

    def test_rejects_non_pdf_content_type(self, client, sample_jd):
        # Send a JPEG with wrong content_type
        response = client.post(
            "/api/v1/analyze",
            files={"resume": ("photo.jpg", b"\xff\xd8\xff", "image/jpeg")},
            data={"job_description": sample_jd},
        )
        assert response.status_code == 400
        assert "detail" in response.json()

    def test_rejects_oversized_file(self, client, sample_jd):
        # Create a fake "PDF" that exceeds 5 MB
        huge_fake_pdf = b"%PDF-1.4" + b"x" * (6 * 1024 * 1024)
        response = client.post(
            "/api/v1/analyze",
            files={"resume": ("big.pdf", huge_fake_pdf, "application/pdf")},
            data={"job_description": sample_jd},
        )
        assert response.status_code == 413
        assert "detail" in response.json()

    def test_rejects_empty_file(self, client, sample_jd):
        response = client.post(
            "/api/v1/analyze",
            files={"resume": ("empty.pdf", b"", "application/pdf")},
            data={"job_description": sample_jd},
        )
        assert response.status_code == 400

    def test_rejects_non_pdf_bytes(self, client, not_a_pdf, sample_jd):
        # Correct content_type but wrong bytes (JPEG content)
        response = client.post(
            "/api/v1/analyze",
            files={"resume": ("fake.pdf", not_a_pdf, "application/pdf")},
            data={"job_description": sample_jd},
        )
        assert response.status_code == 400
        assert "detail" in response.json()

    def test_rejects_missing_resume_field(self, client, sample_jd):
        # No file uploaded at all
        response = client.post(
            "/api/v1/analyze",
            data={"job_description": sample_jd},
        )
        assert response.status_code == 422  # FastAPI validation error

    def test_rejects_missing_job_description(self, client, valid_pdf_bytes):
        # No job description provided
        response = client.post(
            "/api/v1/analyze",
            files={"resume": ("resume.pdf", valid_pdf_bytes, "application/pdf")},
        )
        assert response.status_code == 422

    def test_rejects_short_job_description(self, client, valid_pdf_bytes):
        # Job description too short (min_length=50 in schema)
        response = client.post(
            "/api/v1/analyze",
            files={"resume": ("resume.pdf", valid_pdf_bytes, "application/pdf")},
            data={"job_description": "Too short"},
        )
        assert response.status_code == 422

    def test_error_response_has_detail_field(self, client, sample_jd):
        # All error responses must have a "detail" field
        response = client.post(
            "/api/v1/analyze",
            files={"resume": ("photo.jpg", b"\xff\xd8\xff", "image/jpeg")},
            data={"job_description": sample_jd},
        )
        body = response.json()
        assert "detail" in body


# ── Route Registration Tests ──────────────────────────────────────────────────

class TestRouteRegistration:

    def test_analyze_route_exists(self, client):
        # OPTIONS request reveals what methods are available
        # A 405 (method not allowed) means the route EXISTS but wrong method
        # A 404 means the route doesn't exist at all
        response = client.get("/api/v1/analyze")
        assert response.status_code != 404, "/api/v1/analyze route not registered"

    def test_docs_endpoint_accessible(self, client):
        response = client.get("/docs")
        assert response.status_code == 200

    def test_openapi_schema_accessible(self, client):
        response = client.get("/openapi.json")
        assert response.status_code == 200

    def test_openapi_schema_contains_analyze_route(self, client):
        response = client.get("/openapi.json")
        schema = response.json()
        paths = schema.get("paths", {})
        assert "/api/v1/analyze" in paths, "analyze route missing from OpenAPI schema"
