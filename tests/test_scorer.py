"""
test_scorer.py — Unit Tests for the TF-IDF Scoring Engine
───────────────────────────────────────────────────────────
WHAT WE'RE TESTING:
  1. preprocess_text()       — text normalization behaves correctly
  2. compute_similarity_score() — scores make semantic sense
  3. generate_verdict()      — thresholds produce correct labels
  4. get_score_breakdown()   — output structure is correct

KEY TESTING PHILOSOPHY FOR ML CODE:
  We can't test for exact scores (they depend on vocabulary size and
  document content). Instead we test for ORDERING and RANGE:
    - Identical texts must score higher than unrelated texts
    - Similar texts must score higher than dissimilar texts
    - All scores must stay within [0.0, 1.0]

  This approach is called "property-based testing" and it's the right
  way to test ML outputs where exact values are non-deterministic.
"""

import pytest

from app.services.scorer import (
    compute_similarity_score,
    generate_verdict,
    get_score_breakdown,
    preprocess_text,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def strong_resume() -> str:
    """Resume that closely matches the sample JD."""
    return """
    Senior Python Developer with 4 years of experience.
    Built production REST APIs using FastAPI and Flask.
    Deployed microservices on AWS using Docker and Kubernetes.
    Used PostgreSQL and Redis for data storage and caching.
    Machine learning experience with scikit-learn and PyTorch.
    Practiced test-driven development with pytest.
    Version control with Git and GitHub.
    Agile and scrum methodology.
    """


@pytest.fixture
def weak_resume() -> str:
    """Resume with almost no overlap with the sample JD."""
    return """
    Graphic designer with 5 years of experience in Adobe Photoshop.
    Expert in visual design, typography, and brand identity.
    Proficient in Illustrator, InDesign, and Figma for UI mockups.
    Managed social media campaigns and content creation.
    Photography and video editing skills.
    """


@pytest.fixture
def sample_jd() -> str:
    """A typical Python backend job description."""
    return """
    We are looking for a Python backend engineer.
    Must have experience with FastAPI or Django REST framework.
    Strong knowledge of Docker and AWS required.
    PostgreSQL database experience is essential.
    Familiarity with machine learning and scikit-learn is a plus.
    Must use Git for version control.
    Experience with pytest and test-driven development preferred.
    """


# ── preprocess_text Tests ─────────────────────────────────────────────────────

class TestPreprocessText:

    def test_lowercases_text(self):
        result = preprocess_text("PYTHON FastAPI DOCKER")
        assert result == result.lower()

    def test_removes_punctuation(self):
        result = preprocess_text("Python, FastAPI. Docker!")
        assert "," not in result
        assert "." not in result
        assert "!" not in result

    def test_replaces_hyphens_with_space(self):
        result = preprocess_text("full-stack developer")
        assert "-" not in result
        assert "full" in result
        assert "stack" in result

    def test_collapses_extra_whitespace(self):
        result = preprocess_text("Python    Developer   Engineer")
        assert "  " not in result  # no double spaces

    def test_empty_string_returns_empty(self):
        assert preprocess_text("") == ""

    def test_preserves_tech_terms(self):
        # Critical: tech terms must survive preprocessing
        result = preprocess_text("FastAPI scikit-learn PostgreSQL")
        assert "fastapi" in result
        assert "postgresql" in result
        # scikit-learn becomes "scikit learn" after hyphen replacement
        assert "scikit" in result
        assert "learn" in result


# ── compute_similarity_score Tests ───────────────────────────────────────────

class TestComputeSimilarityScore:

    def test_score_is_float(self, strong_resume, sample_jd):
        score = compute_similarity_score(strong_resume, sample_jd)
        assert isinstance(score, float)

    def test_score_in_valid_range(self, strong_resume, sample_jd):
        score = compute_similarity_score(strong_resume, sample_jd)
        assert 0.0 <= score <= 1.0

    def test_identical_texts_score_high(self, sample_jd):
        # Comparing a document to itself must return 1.0
        score = compute_similarity_score(sample_jd, sample_jd)
        assert score == 1.0

    def test_strong_resume_scores_higher_than_weak(
        self, strong_resume, weak_resume, sample_jd
    ):
        # ORDERING test: the strong resume must outscore the weak one
        strong_score = compute_similarity_score(strong_resume, sample_jd)
        weak_score   = compute_similarity_score(weak_resume, sample_jd)
        assert strong_score > weak_score

    def test_strong_resume_score_is_meaningful(self, strong_resume, sample_jd):
        # NOTE: With only 2 documents, TF-IDF cosine similarity scores compress
        # toward lower values because IDF weights flatten. Scores of 0.15-0.4
        # for a good match are expected in a 2-document corpus.
        # The key property (ordering) is tested in test_strong_resume_scores_higher_than_weak.
        score = compute_similarity_score(strong_resume, sample_jd)
        assert score >= 0.15  # Must be meaningfully above zero

    def test_weak_resume_score_is_low(self, weak_resume, sample_jd):
        # An unrelated resume should score below Strong Fit threshold
        score = compute_similarity_score(weak_resume, sample_jd)
        assert score < 0.65

    def test_score_is_symmetric(self, strong_resume, sample_jd):
        # cosine_similarity(A, B) == cosine_similarity(B, A)
        score_ab = compute_similarity_score(strong_resume, sample_jd)
        score_ba = compute_similarity_score(sample_jd, strong_resume)
        assert abs(score_ab - score_ba) < 0.0001

    def test_empty_resume_raises_value_error(self, sample_jd):
        with pytest.raises(ValueError, match="Resume text is empty"):
            compute_similarity_score("", sample_jd)

    def test_empty_jd_raises_value_error(self, strong_resume):
        with pytest.raises(ValueError, match="Job description text is empty"):
            compute_similarity_score(strong_resume, "")

    def test_score_rounded_to_4_decimal_places(self, strong_resume, sample_jd):
        score = compute_similarity_score(strong_resume, sample_jd)
        # round(x, 4) means at most 4 decimal places
        assert score == round(score, 4)

    def test_completely_unrelated_texts_score_near_zero(self):
        resume = "cats dogs birds animals pet store veterinary"
        jd     = "kubernetes docker microservices devops pipeline ci cd"
        score  = compute_similarity_score(resume, jd)
        assert score < 0.1


# ── generate_verdict Tests ────────────────────────────────────────────────────

class TestGenerateVerdict:

    def test_high_score_is_strong_fit(self):
        # 0.65 is the STRONG_FIT_THRESHOLD in config.py
        assert generate_verdict(0.65) == "Strong Fit"
        assert generate_verdict(0.80) == "Strong Fit"
        assert generate_verdict(1.00) == "Strong Fit"

    def test_mid_score_is_partial_fit(self):
        # Between 0.40 (PARTIAL) and 0.65 (STRONG)
        assert generate_verdict(0.40) == "Partial Fit"
        assert generate_verdict(0.55) == "Partial Fit"
        assert generate_verdict(0.64) == "Partial Fit"

    def test_low_score_is_weak_fit(self):
        # Below 0.40
        assert generate_verdict(0.00) == "Weak Fit"
        assert generate_verdict(0.20) == "Weak Fit"
        assert generate_verdict(0.39) == "Weak Fit"

    def test_verdict_values_are_exact_strings(self):
        # Guard against typos in verdict strings
        valid_verdicts = {"Strong Fit", "Partial Fit", "Weak Fit"}
        for score in [0.0, 0.3, 0.5, 0.7, 1.0]:
            assert generate_verdict(score) in valid_verdicts


# ── get_score_breakdown Tests ─────────────────────────────────────────────────

class TestGetScoreBreakdown:

    def test_returns_all_required_keys(self):
        result = get_score_breakdown(0.75)
        assert "score"      in result
        assert "percentage" in result
        assert "verdict"    in result
        assert "advice"     in result

    def test_percentage_is_score_times_100(self):
        result = get_score_breakdown(0.75)
        assert result["percentage"] == 75.0

    def test_score_preserved(self):
        result = get_score_breakdown(0.55)
        assert result["score"] == 0.55

    def test_verdict_consistent_with_generate_verdict(self):
        for score in [0.2, 0.5, 0.8]:
            breakdown = get_score_breakdown(score)
            assert breakdown["verdict"] == generate_verdict(score)

    def test_advice_is_non_empty_string(self):
        result = get_score_breakdown(0.5)
        assert isinstance(result["advice"], str)
        assert len(result["advice"]) > 0
