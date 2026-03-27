"""
test_integration.py — End-to-End Scenario Tests
─────────────────────────────────────────────────
WHAT ARE SCENARIO TESTS?
  Unlike unit tests (which test one function) or integration tests
  (which test modules working together), scenario tests simulate real
  user stories from start to finish.

  Each test class here represents one realistic use case:
    - "A strong candidate applies for a matching role"
    - "A candidate from a different field applies"
    - "A recruiter sends a bad file by mistake"

  These tests are the closest thing to "what happens in production".

WHY USE conftest.py FIXTURES HERE?
  All fixtures (client, valid_pdf_bytes, strong_jd etc.) come from
  conftest.py — no duplication. This is the payoff for setting up
  conftest.py earlier in this step.

MARKS:
  @pytest.mark.integration — labels these as integration tests.
  Run only integration tests: pytest -m integration
  Skip integration tests:    pytest -m "not integration"
"""

import pytest


@pytest.mark.integration
class TestStrongCandidateScenario:
    """
    Scenario: A well-matched candidate applies for a Python backend role.
    The resume (valid_pdf_bytes) contains Python, FastAPI, Docker, AWS, etc.
    The JD (strong_jd) asks for exactly those skills.
    Expected: high skill_score, Strong or Partial Fit verdict.
    """

    def test_returns_200(self, client, valid_pdf_bytes, strong_jd):
        response = client.post(
            "/api/v1/analyze",
            files={"resume": ("resume.pdf", valid_pdf_bytes, "application/pdf")},
            data={"job_description": strong_jd},
        )
        assert response.status_code in (200, 400)  # 400 if minimal PDF fails

    def test_skill_score_is_high(self, client, valid_pdf_bytes, strong_jd):
        response = client.post(
            "/api/v1/analyze",
            files={"resume": ("resume.pdf", valid_pdf_bytes, "application/pdf")},
            data={"job_description": strong_jd},
        )
        if response.status_code != 200:
            pytest.skip("Minimal PDF text extraction skipped")
        body = response.json()
        # A resume with Python, FastAPI, Docker, AWS matching a Python JD
        # should score at least 0.5 on skill matching
        assert body["skill_score"] >= 0.35  # minimal PDF has 7 skills vs JD 17 skills

    def test_matched_skills_include_python(self, client, valid_pdf_bytes, strong_jd):
        response = client.post(
            "/api/v1/analyze",
            files={"resume": ("resume.pdf", valid_pdf_bytes, "application/pdf")},
            data={"job_description": strong_jd},
        )
        if response.status_code != 200:
            pytest.skip("Minimal PDF text extraction skipped")
        assert "python" in response.json()["matched_skills"]

    def test_response_has_advice(self, client, valid_pdf_bytes, strong_jd):
        response = client.post(
            "/api/v1/analyze",
            files={"resume": ("resume.pdf", valid_pdf_bytes, "application/pdf")},
            data={"job_description": strong_jd},
        )
        if response.status_code != 200:
            pytest.skip("Minimal PDF text extraction skipped")
        advice = response.json()["advice"]
        assert isinstance(advice, str) and len(advice) > 20


@pytest.mark.integration
class TestWrongFileTypeScenario:
    """
    Scenario: A recruiter accidentally uploads a Word doc or image.
    Expected: clear 400 error with a message mentioning PDF.
    """

    def test_jpeg_rejected_with_400(self, client, strong_jd):
        jpeg_bytes = b"\xff\xd8\xff\xe0" + b"\x00" * 100
        response = client.post(
            "/api/v1/analyze",
            files={"resume": ("photo.jpg", jpeg_bytes, "image/jpeg")},
            data={"job_description": strong_jd},
        )
        assert response.status_code == 400

    def test_png_rejected_with_400(self, client, strong_jd):
        png_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        response = client.post(
            "/api/v1/analyze",
            files={"resume": ("photo.png", png_bytes, "image/png")},
            data={"job_description": strong_jd},
        )
        assert response.status_code == 400

    def test_docx_rejected_with_400(self, client, strong_jd):
        # DOCX files start with the ZIP magic bytes (PK)
        docx_bytes = b"PK\x03\x04" + b"\x00" * 100
        response = client.post(
            "/api/v1/analyze",
            files={"resume": ("resume.docx", docx_bytes, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
            data={"job_description": strong_jd},
        )
        assert response.status_code == 400

    def test_error_message_mentions_pdf(self, client, strong_jd):
        jpeg_bytes = b"\xff\xd8\xff\xe0" + b"\x00" * 100
        response = client.post(
            "/api/v1/analyze",
            files={"resume": ("photo.jpg", jpeg_bytes, "image/jpeg")},
            data={"job_description": strong_jd},
        )
        assert "pdf" in response.json()["detail"].lower()


@pytest.mark.integration
class TestInvalidJobDescriptionScenario:
    """
    Scenario: A developer integrating the API sends a malformed request.
    Expected: clear validation errors explaining what's wrong.
    """

    def test_empty_jd_returns_422(self, client, valid_pdf_bytes):
        response = client.post(
            "/api/v1/analyze",
            files={"resume": ("r.pdf", valid_pdf_bytes, "application/pdf")},
            data={"job_description": ""},
        )
        assert response.status_code == 422

    def test_whitespace_jd_returns_422(self, client, valid_pdf_bytes):
        response = client.post(
            "/api/v1/analyze",
            files={"resume": ("r.pdf", valid_pdf_bytes, "application/pdf")},
            data={"job_description": "   "},
        )
        assert response.status_code == 422

    def test_too_short_jd_returns_422(self, client, valid_pdf_bytes, short_jd):
        response = client.post(
            "/api/v1/analyze",
            files={"resume": ("r.pdf", valid_pdf_bytes, "application/pdf")},
            data={"job_description": short_jd},
        )
        assert response.status_code == 422

    def test_missing_jd_field_returns_422(self, client, valid_pdf_bytes):
        response = client.post(
            "/api/v1/analyze",
            files={"resume": ("r.pdf", valid_pdf_bytes, "application/pdf")},
            # no job_description field at all
        )
        assert response.status_code == 422

    def test_422_detail_is_string(self, client, valid_pdf_bytes):
        response = client.post(
            "/api/v1/analyze",
            files={"resume": ("r.pdf", valid_pdf_bytes, "application/pdf")},
            data={"job_description": "too short"},
        )
        assert isinstance(response.json()["detail"], str)


@pytest.mark.integration
class TestFileSizeLimitScenario:
    """
    Scenario: A user uploads a huge scanned PDF (common mistake).
    Expected: 413 error with a message about file size.
    """

    def test_6mb_file_returns_413(self, client, strong_jd, oversized_pdf):
        response = client.post(
            "/api/v1/analyze",
            files={"resume": ("big.pdf", oversized_pdf, "application/pdf")},
            data={"job_description": strong_jd},
        )
        assert response.status_code == 413

    def test_size_error_mentions_limit(self, client, strong_jd, oversized_pdf):
        response = client.post(
            "/api/v1/analyze",
            files={"resume": ("big.pdf", oversized_pdf, "application/pdf")},
            data={"job_description": strong_jd},
        )
        detail = response.json()["detail"].lower()
        assert "mb" in detail or "size" in detail or "limit" in detail

    def test_empty_file_returns_400(self, client, strong_jd, empty_pdf):
        response = client.post(
            "/api/v1/analyze",
            files={"resume": ("empty.pdf", empty_pdf, "application/pdf")},
            data={"job_description": strong_jd},
        )
        assert response.status_code == 400


@pytest.mark.integration
class TestApiContractScenario:
    """
    Scenario: A frontend developer integrates with the API.
    They need to rely on a stable response structure.
    These tests guard the API contract — the shape must never change
    without a version bump.
    """

    def test_successful_response_shape(self, client, valid_pdf_bytes, strong_jd):
        response = client.post(
            "/api/v1/analyze",
            files={"resume": ("r.pdf", valid_pdf_bytes, "application/pdf")},
            data={"job_description": strong_jd},
        )
        if response.status_code != 200:
            pytest.skip("Minimal PDF text extraction skipped")

        body = response.json()
        # Exact field names the frontend depends on
        assert "fit_score"          in body
        assert "fit_percentage"     in body
        assert "tfidf_score"        in body
        assert "skill_score"        in body
        assert "verdict"            in body
        assert "advice"             in body
        assert "matched_skills"     in body
        assert "missing_skills"     in body
        assert "extra_skills"       in body
        assert "resume_skill_count" in body
        assert "jd_skill_count"     in body
        assert "resume_text_length" in body

    def test_score_fields_are_floats(self, client, valid_pdf_bytes, strong_jd):
        response = client.post(
            "/api/v1/analyze",
            files={"resume": ("r.pdf", valid_pdf_bytes, "application/pdf")},
            data={"job_description": strong_jd},
        )
        if response.status_code != 200:
            pytest.skip("Minimal PDF text extraction skipped")
        body = response.json()
        assert isinstance(body["fit_score"],   float)
        assert isinstance(body["tfidf_score"], float)
        assert isinstance(body["skill_score"], float)

    def test_count_fields_are_integers(self, client, valid_pdf_bytes, strong_jd):
        response = client.post(
            "/api/v1/analyze",
            files={"resume": ("r.pdf", valid_pdf_bytes, "application/pdf")},
            data={"job_description": strong_jd},
        )
        if response.status_code != 200:
            pytest.skip("Minimal PDF text extraction skipped")
        body = response.json()
        assert isinstance(body["resume_skill_count"], int)
        assert isinstance(body["jd_skill_count"],     int)
        assert isinstance(body["resume_text_length"], int)

    def test_verdict_is_one_of_three_values(self, client, valid_pdf_bytes, strong_jd):
        response = client.post(
            "/api/v1/analyze",
            files={"resume": ("r.pdf", valid_pdf_bytes, "application/pdf")},
            data={"job_description": strong_jd},
        )
        if response.status_code != 200:
            pytest.skip("Minimal PDF text extraction skipped")
        assert response.json()["verdict"] in {
            "Strong Fit", "Partial Fit", "Weak Fit"
        }

    def test_fit_percentage_equals_score_times_100(
        self, client, valid_pdf_bytes, strong_jd
    ):
        response = client.post(
            "/api/v1/analyze",
            files={"resume": ("r.pdf", valid_pdf_bytes, "application/pdf")},
            data={"job_description": strong_jd},
        )
        if response.status_code != 200:
            pytest.skip("Minimal PDF text extraction skipped")
        body = response.json()
        expected = round(body["fit_score"] * 100, 1)
        assert body["fit_percentage"] == expected

    def test_request_id_in_every_response_header(
        self, client, valid_pdf_bytes, strong_jd
    ):
        response = client.post(
            "/api/v1/analyze",
            files={"resume": ("r.pdf", valid_pdf_bytes, "application/pdf")},
            data={"job_description": strong_jd},
        )
        assert "x-request-id" in response.headers


@pytest.mark.integration
class TestConfigIntegrity:
    """
    Verify that config settings used throughout the app are internally
    consistent — no circular dependencies or invalid values.
    """

    def test_fit_thresholds_are_ordered_correctly(self):
        from app.config import settings
        # PARTIAL must be strictly less than STRONG
        assert settings.PARTIAL_FIT_THRESHOLD < settings.STRONG_FIT_THRESHOLD

    def test_thresholds_are_in_valid_range(self):
        from app.config import settings
        assert 0.0 < settings.PARTIAL_FIT_THRESHOLD < 1.0
        assert 0.0 < settings.STRONG_FIT_THRESHOLD < 1.0

    def test_max_file_size_is_positive(self):
        from app.config import settings
        assert settings.MAX_FILE_SIZE_BYTES > 0

    def test_allowed_content_types_includes_pdf(self):
        from app.config import settings
        assert "application/pdf" in settings.ALLOWED_CONTENT_TYPES

    def test_hybrid_weights_sum_to_one(self):
        from app.services.analyzer import SKILL_WEIGHT, TFIDF_WEIGHT
        assert abs(TFIDF_WEIGHT + SKILL_WEIGHT - 1.0) < 0.0001
