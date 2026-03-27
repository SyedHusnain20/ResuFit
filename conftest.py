"""
conftest.py — Shared Pytest Fixtures
──────────────────────────────────────
WHAT IS conftest.py?
  A special pytest file that is automatically discovered and loaded before
  any tests run. Fixtures defined here are available to ALL test files in
  the project without needing to import them explicitly.

WHY MOVE FIXTURES HERE?
  Currently, each test file defines its own `valid_pdf_bytes`, `sample_jd`,
  and `not_a_pdf` fixtures — duplicated across 4 files. If the PDF structure
  changes, you'd need to update 4 files. conftest.py is the single source
  of truth for shared test data.

FIXTURE SCOPES:
  scope="function"  — recreated for every test (default)
  scope="module"    — created once per test file, shared across tests in it
  scope="session"   — created once for the entire test run

  We use "session" for heavy fixtures (PDF bytes, JD strings) that are
  expensive to create and don't mutate between tests.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app


# ── HTTP Client ───────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def client() -> TestClient:
    """
    A FastAPI TestClient shared across the entire test session.
    scope="session" means one client for ALL tests — efficient and fast.
    raise_server_exceptions=False lets us assert on 500 responses
    instead of having them re-raise as Python exceptions in tests.
    """
    return TestClient(app, raise_server_exceptions=False)


# ── PDF Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def valid_pdf_bytes() -> bytes:
    """
    A real, minimal, parseable PDF containing tech skill keywords.
    Used across multiple test files — defined once here.

    The text embedded is: "Python FastAPI Docker AWS PostgreSQL Developer Git pytest"
    This ensures the skill extractor finds meaningful skills in tests.
    """
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


@pytest.fixture(scope="session")
def not_a_pdf() -> bytes:
    """JPEG magic bytes — clearly not a PDF. Used to test rejection logic."""
    return b"\xff\xd8\xff\xe0" + b"\x00" * 200


@pytest.fixture(scope="session")
def empty_pdf() -> bytes:
    """Zero bytes — used to test empty file rejection."""
    return b""


@pytest.fixture(scope="session")
def oversized_pdf() -> bytes:
    """
    A fake 'PDF' that exceeds the 5 MB size limit.
    Used to test the file size validation guard.
    """
    return b"%PDF-1.4" + b"x" * (6 * 1024 * 1024)


# ── Job Description Fixtures ──────────────────────────────────────────────────

@pytest.fixture(scope="session")
def strong_jd() -> str:
    """
    A realistic Python backend JD that closely matches the dummy resume.
    Expects high skill_score when tested against valid_pdf_bytes.
    """
    return """
    We are hiring a Python Backend Engineer to join our growing team.
    Must have experience with FastAPI or Django REST framework.
    Strong knowledge of Docker and AWS is required for our cloud infrastructure.
    You must be comfortable working with PostgreSQL databases at scale.
    Experience with machine learning pipelines using scikit-learn is a plus.
    We use Git and GitHub Actions for version control and continuous integration.
    pytest knowledge required. Agile environment with bi-weekly sprints.
    Must understand REST API design principles and microservices architecture.
    """


@pytest.fixture(scope="session")
def unrelated_jd() -> str:
    """
    A JD completely unrelated to software engineering.
    Should produce a low score against any developer resume.
    """
    return """
    We are looking for an experienced Chef de Cuisine to lead our kitchen team.
    Must have culinary arts degree and 5 years of fine dining experience.
    Expert knowledge of French cuisine, knife skills, and menu planning required.
    Experience managing kitchen staff and food inventory essential.
    Strong understanding of food safety regulations and HACCP certification needed.
    Passion for seasonal ingredients and farm-to-table cooking philosophy preferred.
    """


@pytest.fixture(scope="session")
def minimal_jd() -> str:
    """Exactly 50 characters — right at the minimum_length boundary."""
    return "Python developer needed with FastAPI and Docker exp"


@pytest.fixture(scope="session")
def short_jd() -> str:
    """Under 50 characters — should be rejected with 422."""
    return "Python dev needed"


# ── Resume Text Fixtures (raw text, not PDF) ──────────────────────────────────

@pytest.fixture(scope="session")
def strong_resume_text() -> str:
    """Full resume text matching the strong_jd closely."""
    return """
    Senior Python Developer with 4 years of professional experience.
    Built production REST APIs using FastAPI and Flask.
    Deployed containerized microservices on AWS using Docker and Kubernetes.
    Strong database skills with PostgreSQL and Redis.
    Applied machine learning models using scikit-learn and PyTorch.
    Practiced test-driven development using pytest and GitHub Actions CI/CD.
    Experienced with Git, agile, and scrum methodologies.
    """


@pytest.fixture(scope="session")
def weak_resume_text() -> str:
    """Resume text with minimal overlap with any tech JD."""
    return """
    Experienced graphic designer with 6 years in visual communication.
    Expert in Adobe Photoshop, Illustrator, and InDesign.
    Strong portfolio of brand identity and logo design work.
    Experience in print media, packaging design, and typography.
    """
