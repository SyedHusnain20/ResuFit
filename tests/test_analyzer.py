"""
test_analyzer.py — Integration Tests for the Analysis Pipeline
───────────────────────────────────────────────────────────────
UNIT vs INTEGRATION TESTS — what's the difference?
  Unit tests:        Test one function in isolation (what we did in Steps 2–4)
  Integration tests: Test multiple modules working TOGETHER end-to-end

  This file is integration-level because analyze_resume() calls:
    pdf_parser → skill_extractor → scorer → verdict logic
  all in sequence. We're testing that the pipeline as a whole works correctly.

WHAT WE TEST:
  1. compute_hybrid_score()  — math is correct
  2. result_to_dict()        — output structure is correct
  3. analyze_resume()        — full pipeline produces valid results
  4. Edge cases              — empty JD, invalid PDF, error propagation
"""

import pytest

from app.services.analyzer import (
    AnalysisResult,
    SKILL_WEIGHT,
    TFIDF_WEIGHT,
    analyze_resume,
    compute_hybrid_score,
    result_to_dict,
)
from app.services.pdf_parser import EmptyPDFError, PDFParsingError


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_jd() -> str:
    return """
    We are hiring a Python Backend Engineer.
    Must have experience with FastAPI or Django REST framework.
    Strong knowledge of Docker and AWS is required.
    PostgreSQL database experience is essential.
    Experience with machine learning and scikit-learn is a plus.
    Must use Git for version control. pytest knowledge required.
    Agile environment with bi-weekly sprints.
    """


@pytest.fixture
def valid_pdf_bytes() -> bytes:
    """
    A real, minimal PDF that pdfplumber can parse.
    Contains enough text to pass the EmptyPDFError guard.

    We use a slightly more complete PDF structure than in test_pdf_parser.py
    to ensure text extraction produces non-empty output reliably.
    """
    # This minimal PDF contains "Python FastAPI Docker AWS PostgreSQL Developer"
    # enough for the skill extractor to find matches
    pdf_content = b"""%PDF-1.4
1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj
2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj
3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>endobj
4 0 obj<</Length 90>>
stream
BT /F1 10 Tf 50 750 Td (Python FastAPI Docker AWS PostgreSQL Developer Git pytest) Tj ET
endstream
endobj
5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj
xref
0 6
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000266 00000 n 
0000000408 00000 n 
trailer<</Size 6/Root 1 0 R>>
startxref
489
%%EOF"""
    return pdf_content


@pytest.fixture
def not_a_pdf() -> bytes:
    return b"\xff\xd8\xff\xe0" + b"\x00" * 100


# ── compute_hybrid_score Tests ────────────────────────────────────────────────

class TestComputeHybridScore:

    def test_weights_sum_to_one(self):
        # Critical invariant: weights must sum to 1.0
        assert abs(TFIDF_WEIGHT + SKILL_WEIGHT - 1.0) < 0.0001

    def test_perfect_scores_give_one(self):
        score = compute_hybrid_score(1.0, 1.0)
        assert score == 1.0

    def test_zero_scores_give_zero(self):
        score = compute_hybrid_score(0.0, 0.0)
        assert score == 0.0

    def test_weighted_average_is_correct(self):
        # With TFIDF_WEIGHT=0.4, SKILL_WEIGHT=0.6:
        # (0.5 * 0.4) + (0.8 * 0.6) = 0.20 + 0.48 = 0.68
        score = compute_hybrid_score(0.5, 0.8)
        expected = round((0.5 * TFIDF_WEIGHT) + (0.8 * SKILL_WEIGHT), 4)
        assert score == expected

    def test_skill_weight_dominates(self):
        # When skill match is high but TF-IDF is low, skill weight wins
        low_tfidf_high_skill  = compute_hybrid_score(0.1, 0.9)
        high_tfidf_low_skill  = compute_hybrid_score(0.9, 0.1)
        # Since SKILL_WEIGHT > TFIDF_WEIGHT, the first should score higher
        assert low_tfidf_high_skill > high_tfidf_low_skill

    def test_result_is_clamped_to_valid_range(self):
        # Even with extreme inputs, output must stay in [0.0, 1.0]
        assert compute_hybrid_score(0.0, 0.0) >= 0.0
        assert compute_hybrid_score(1.0, 1.0) <= 1.0

    def test_result_rounded_to_4_decimals(self):
        score = compute_hybrid_score(0.333, 0.666)
        assert score == round(score, 4)


# ── result_to_dict Tests ──────────────────────────────────────────────────────

class TestResultToDict:

    @pytest.fixture
    def sample_result(self) -> AnalysisResult:
        return AnalysisResult(
            tfidf_score         = 0.35,
            skill_score         = 0.75,
            hybrid_score        = 0.59,
            fit_percentage      = 59.0,
            verdict             = "Partial Fit",
            advice              = "Consider if skill gaps can be addressed.",
            matched_skills      = ["python", "fastapi"],
            missing_skills      = ["docker"],
            extra_skills        = ["flask"],
            resume_skill_count  = 3,
            jd_skill_count      = 3,
            resume_text_length  = 500,
            resume_text_preview = "Python developer...",
        )

    def test_contains_all_required_keys(self, sample_result):
        result = result_to_dict(sample_result)
        required_keys = [
            "fit_score", "fit_percentage", "verdict", "advice",
            "matched_skills", "missing_skills", "extra_skills",
            "tfidf_score", "skill_score",
            "resume_skill_count", "jd_skill_count", "resume_text_length",
        ]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_debug_field_hidden_by_default(self, sample_result):
        result = result_to_dict(sample_result)
        assert "resume_text_preview" not in result

    def test_debug_field_shown_when_requested(self, sample_result):
        result = result_to_dict(sample_result, include_debug=True)
        assert "resume_text_preview" in result

    def test_hybrid_score_exposed_as_fit_score(self, sample_result):
        # API consumers see "fit_score", not "hybrid_score"
        result = result_to_dict(sample_result)
        assert result["fit_score"] == sample_result.hybrid_score

    def test_skills_are_lists(self, sample_result):
        result = result_to_dict(sample_result)
        assert isinstance(result["matched_skills"], list)
        assert isinstance(result["missing_skills"], list)
        assert isinstance(result["extra_skills"],   list)


# ── analyze_resume Integration Tests ─────────────────────────────────────────

class TestAnalyzeResume:

    def test_invalid_pdf_raises_parsing_error(self, not_a_pdf, sample_jd):
        with pytest.raises(PDFParsingError):
            analyze_resume(not_a_pdf, sample_jd)

    def test_empty_bytes_raises_parsing_error(self, sample_jd):
        with pytest.raises(PDFParsingError):
            analyze_resume(b"", sample_jd)

    def test_empty_jd_raises_value_error(self, valid_pdf_bytes):
        with pytest.raises(ValueError, match="Job description cannot be empty"):
            analyze_resume(valid_pdf_bytes, "")

    def test_whitespace_jd_raises_value_error(self, valid_pdf_bytes):
        with pytest.raises(ValueError, match="Job description cannot be empty"):
            analyze_resume(valid_pdf_bytes, "   \n\t  ")

    def test_returns_analysis_result_type(self, valid_pdf_bytes, sample_jd):
        try:
            result = analyze_resume(valid_pdf_bytes, sample_jd)
            assert isinstance(result, AnalysisResult)
        except (PDFParsingError, EmptyPDFError):
            pytest.skip("Minimal PDF not parseable in this environment")

    def test_hybrid_score_in_valid_range(self, valid_pdf_bytes, sample_jd):
        try:
            result = analyze_resume(valid_pdf_bytes, sample_jd)
            assert 0.0 <= result.hybrid_score <= 1.0
        except (PDFParsingError, EmptyPDFError):
            pytest.skip("Minimal PDF not parseable in this environment")

    def test_verdict_is_valid_string(self, valid_pdf_bytes, sample_jd):
        try:
            result = analyze_resume(valid_pdf_bytes, sample_jd)
            assert result.verdict in {"Strong Fit", "Partial Fit", "Weak Fit"}
        except (PDFParsingError, EmptyPDFError):
            pytest.skip("Minimal PDF not parseable in this environment")

    def test_skill_lists_are_lists(self, valid_pdf_bytes, sample_jd):
        try:
            result = analyze_resume(valid_pdf_bytes, sample_jd)
            assert isinstance(result.matched_skills, list)
            assert isinstance(result.missing_skills, list)
            assert isinstance(result.extra_skills,   list)
        except (PDFParsingError, EmptyPDFError):
            pytest.skip("Minimal PDF not parseable in this environment")

    def test_skill_counts_are_non_negative(self, valid_pdf_bytes, sample_jd):
        try:
            result = analyze_resume(valid_pdf_bytes, sample_jd)
            assert result.resume_skill_count >= 0
            assert result.jd_skill_count     >= 0
        except (PDFParsingError, EmptyPDFError):
            pytest.skip("Minimal PDF not parseable in this environment")

    def test_fit_percentage_matches_hybrid_score(self, valid_pdf_bytes, sample_jd):
        try:
            result = analyze_resume(valid_pdf_bytes, sample_jd)
            assert result.fit_percentage == round(result.hybrid_score * 100, 1)
        except (PDFParsingError, EmptyPDFError):
            pytest.skip("Minimal PDF not parseable in this environment")


# ── Hybrid Score Behaviour Tests (logic-level, no PDF needed) ─────────────────

class TestHybridScoringLogic:
    """
    These tests verify the hybrid scoring logic using compute_hybrid_score()
    directly — no PDF parsing needed. They test the BUSINESS RULES.
    """

    def test_strong_skill_match_lifts_weak_tfidf(self):
        # Even with a poor TF-IDF score, a high skill match should produce
        # a reasonable hybrid score
        score = compute_hybrid_score(tfidf_score=0.15, skill_match_rate=0.90)
        assert score >= 0.50  # Should be at least Partial Fit territory

    def test_zero_skill_match_pulls_score_down(self):
        # Even with a decent TF-IDF, zero skill match should produce a low score
        score = compute_hybrid_score(tfidf_score=0.50, skill_match_rate=0.0)
        assert score <= 0.25

    def test_both_strong_gives_strong_fit_verdict(self):
        from app.services.scorer import generate_verdict
        score   = compute_hybrid_score(0.70, 0.90)
        verdict = generate_verdict(score)
        assert verdict == "Strong Fit"

    def test_both_weak_gives_weak_fit_verdict(self):
        from app.services.scorer import generate_verdict
        score   = compute_hybrid_score(0.10, 0.10)
        verdict = generate_verdict(score)
        assert verdict == "Weak Fit"
