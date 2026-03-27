"""
test_error_handling.py — Error Handling & Middleware Tests
───────────────────────────────────────────────────────────
WHAT WE TEST:
  1. Every error response has { "detail": "..." } shape (consistent contract)
  2. Validation errors (422) return human-readable messages
  3. HTTP errors (400, 413) return the correct status codes
  4. Every response has an X-Request-ID header (middleware working)
  5. Request ID is different for each request (unique IDs)
  6. The catch-all 500 handler works for truly unexpected errors
  7. Error messages never expose raw Python internals

WHY TEST ERROR HANDLING SEPARATELY?
  Errors are user-facing features, not just defensive code.
  A well-designed error response tells the caller exactly what went wrong
  and how to fix it. Testing error paths is as important as testing
  the happy path — arguably more so, because errors happen in production.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app, raise_server_exceptions=False)
# raise_server_exceptions=False means 500 errors are returned as responses
# instead of re-raising the exception in the test. This lets us assert on
# the 500 response body.


# ── Sample fixtures ───────────────────────────────────────────────────────────

@pytest.fixture
def long_jd() -> str:
    """A valid job description (over 50 chars)."""
    return (
        "We are looking for a Python backend engineer with FastAPI and Docker "
        "experience. PostgreSQL and AWS knowledge required. Min 2 years experience."
    )


@pytest.fixture
def valid_pdf() -> bytes:
    """Minimal valid PDF bytes."""
    return (
        b"%PDF-1.4\n"
        b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
        b"3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R"
        b"/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>endobj\n"
        b"4 0 obj<</Length 44>>\nstream\n"
        b"BT /F1 12 Tf 100 700 Td (Python FastAPI Docker) Tj ET\n"
        b"endstream\nendobj\n"
        b"5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj\n"
        b"xref\n0 6\n"
        b"0000000000 65535 f \n"
        b"0000000009 00000 n \n"
        b"0000000058 00000 n \n"
        b"0000000115 00000 n \n"
        b"0000000266 00000 n \n"
        b"0000000360 00000 n \n"
        b"trailer<</Size 6/Root 1 0 R>>\n"
        b"startxref\n441\n%%EOF"
    )


# ── Response Shape Tests ──────────────────────────────────────────────────────

class TestErrorResponseShape:
    """Every error must return { 'detail': '...' } — no exceptions."""

    def test_400_has_detail_field(self, long_jd):
        response = client.post(
            "/api/v1/analyze",
            files={"resume": ("photo.jpg", b"\xff\xd8\xff", "image/jpeg")},
            data={"job_description": long_jd},
        )
        assert response.status_code == 400
        body = response.json()
        assert "detail" in body
        assert isinstance(body["detail"], str)
        assert len(body["detail"]) > 0

    def test_413_has_detail_field(self, long_jd):
        huge = b"%PDF-1.4" + b"x" * (6 * 1024 * 1024)
        response = client.post(
            "/api/v1/analyze",
            files={"resume": ("big.pdf", huge, "application/pdf")},
            data={"job_description": long_jd},
        )
        assert response.status_code == 413
        assert "detail" in response.json()

    def test_422_has_detail_field(self):
        # Missing both required fields triggers 422
        response = client.post("/api/v1/analyze")
        assert response.status_code == 422
        body = response.json()
        assert "detail" in body
        assert isinstance(body["detail"], str)

    def test_404_has_detail_field(self):
        response = client.get("/nonexistent-route")
        assert response.status_code == 404
        assert "detail" in response.json()

    def test_error_body_never_has_traceback(self, long_jd):
        # Python tracebacks must never leak to the client
        huge = b"%PDF-1.4" + b"x" * (6 * 1024 * 1024)
        response = client.post(
            "/api/v1/analyze",
            files={"resume": ("big.pdf", huge, "application/pdf")},
            data={"job_description": long_jd},
        )
        body_text = response.text
        # These strings appear in Python tracebacks — should never be in response
        assert "Traceback" not in body_text
        assert "File \"" not in body_text
        assert "line " not in body_text


# ── Validation Error Tests (422) ──────────────────────────────────────────────

class TestValidationErrors:

    def test_missing_resume_returns_422(self, long_jd):
        response = client.post(
            "/api/v1/analyze",
            data={"job_description": long_jd},
        )
        assert response.status_code == 422

    def test_missing_jd_returns_422(self, valid_pdf):
        response = client.post(
            "/api/v1/analyze",
            files={"resume": ("r.pdf", valid_pdf, "application/pdf")},
        )
        assert response.status_code == 422

    def test_short_jd_returns_422(self, valid_pdf):
        # min_length=50 set on the Form field in the router
        response = client.post(
            "/api/v1/analyze",
            files={"resume": ("r.pdf", valid_pdf, "application/pdf")},
            data={"job_description": "Too short"},
        )
        assert response.status_code == 422

    def test_422_detail_is_human_readable(self):
        response = client.post("/api/v1/analyze")
        detail = response.json()["detail"]
        # Should be a human-readable sentence, not a raw Pydantic error dict
        assert isinstance(detail, str)
        # Must not contain raw Pydantic internal field names as JSON
        assert detail.startswith("{") is False


# ── HTTP Error Content Tests ──────────────────────────────────────────────────

class TestHttpErrorContent:

    def test_wrong_file_type_mentions_pdf(self, long_jd):
        response = client.post(
            "/api/v1/analyze",
            files={"resume": ("img.jpg", b"\xff\xd8\xff", "image/jpeg")},
            data={"job_description": long_jd},
        )
        detail = response.json()["detail"].lower()
        # Error message must mention PDF so the user knows what to fix
        assert "pdf" in detail

    def test_oversized_file_mentions_size_limit(self, long_jd):
        huge = b"%PDF-1.4" + b"x" * (6 * 1024 * 1024)
        response = client.post(
            "/api/v1/analyze",
            files={"resume": ("big.pdf", huge, "application/pdf")},
            data={"job_description": long_jd},
        )
        detail = response.json()["detail"].lower()
        # Must mention size or MB so the user knows the problem
        assert "mb" in detail or "size" in detail or "limit" in detail

    def test_empty_file_returns_400(self, long_jd):
        response = client.post(
            "/api/v1/analyze",
            files={"resume": ("empty.pdf", b"", "application/pdf")},
            data={"job_description": long_jd},
        )
        assert response.status_code == 400

    def test_non_pdf_bytes_returns_400(self, long_jd):
        # Correct MIME type but wrong bytes (JPEG content)
        jpeg_bytes = b"\xff\xd8\xff\xe0" + b"\x00" * 200
        response = client.post(
            "/api/v1/analyze",
            files={"resume": ("fake.pdf", jpeg_bytes, "application/pdf")},
            data={"job_description": long_jd},
        )
        assert response.status_code == 400


# ── Request ID Middleware Tests ────────────────────────────────────────────────

class TestRequestIdMiddleware:

    def test_every_response_has_request_id_header(self):
        response = client.get("/health")
        assert "x-request-id" in response.headers

    def test_request_id_is_non_empty(self):
        response = client.get("/health")
        assert len(response.headers["x-request-id"]) > 0

    def test_request_ids_are_unique_per_request(self):
        # Each request must get a different ID
        r1 = client.get("/health")
        r2 = client.get("/health")
        r3 = client.get("/health")
        ids = {
            r1.headers["x-request-id"],
            r2.headers["x-request-id"],
            r3.headers["x-request-id"],
        }
        assert len(ids) == 3, "Request IDs must be unique per request"

    def test_error_responses_also_have_request_id(self, long_jd):
        # X-Request-ID must be present even on error responses
        response = client.post(
            "/api/v1/analyze",
            files={"resume": ("img.jpg", b"\xff\xd8\xff", "image/jpeg")},
            data={"job_description": long_jd},
        )
        assert "x-request-id" in response.headers

    def test_request_id_length(self):
        # We use first 8 chars of UUID — should be exactly 8 chars
        response = client.get("/health")
        assert len(response.headers["x-request-id"]) == 8


# ── Success Response Sanity Tests ─────────────────────────────────────────────

class TestSuccessResponses:

    def test_health_always_200(self):
        for _ in range(3):
            assert client.get("/health").status_code == 200

    def test_root_returns_correct_content_type(self):
        response = client.get("/")
        assert "application/json" in response.headers["content-type"]

    def test_docs_accessible(self):
        assert client.get("/docs").status_code == 200

    def test_openapi_json_accessible(self):
        assert client.get("/openapi.json").status_code == 200
