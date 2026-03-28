"""
analyzer.py — Main Resume Analysis Pipeline
─────────────────────────────────────────────
RESPONSIBILITY:
  This is the orchestrator — the single entry point for all analysis.
  It calls the three service modules in sequence and combines their
  outputs into one structured result dictionary.

WHY A SEPARATE ORCHESTRATOR MODULE?
  Without this, the router (in Step 6) would need to know about PDF parsing,
  skill extraction, AND scoring — violating the single responsibility principle.
  With this module, the router calls ONE function and gets ONE result back.
  The router doesn't care how the analysis works — only what it returns.

THE PIPELINE (in order):
  1. validate_pdf_bytes()       — reject non-PDFs early
  2. extract_text_from_pdf()    — get raw text from PDF bytes
  3. extract_skills(resume)     — find skills in resume text
  4. extract_skills(jd)         — find skills in job description text
  5. compare_skills()           — diff the two skill sets
  6. compute_similarity_score() — TF-IDF cosine similarity
  7. compute_hybrid_score()     — combine TF-IDF + skill match rate
  8. generate_verdict()         — Strong / Partial / Weak Fit
  9. Assemble and return result dict

THE HYBRID SCORING MODEL:
  We combine two independent signals with weighted averaging:

    hybrid_score = (tfidf_score × TFIDF_WEIGHT) + (skill_rate × SKILL_WEIGHT)

  Default weights: TF-IDF = 40%, Skill Match Rate = 60%

  WHY 60% FOR SKILLS?
    For technical roles, specific skills ("Docker", "FastAPI", "PostgreSQL")
    are more predictive of job fit than overall writing similarity.
    A candidate who lists the exact required technologies is a better match
    than one whose resume "sounds like" the JD but lacks the specific tools.

  WHY KEEP TF-IDF AT ALL?
    TF-IDF captures context that skills alone miss:
    - Seniority language ("led a team of", "architected", "junior developer")
    - Domain vocabulary ("distributed systems", "real-time processing")
    - Soft skill language ("cross-functional", "stakeholder communication")
    These don't map to discrete skills but matter for overall fit.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

from app.services.pdf_parser import (
    EmptyPDFError,
    PDFParsingError,
    extract_text_from_pdf,
    validate_pdf_bytes,
)
from app.services.scorer import (
    compute_similarity_score,
    generate_verdict,
    get_score_breakdown,
)
from app.services.skill_extractor import compare_skills, extract_skills

logger = logging.getLogger(__name__)


# ── Weights for hybrid scoring ────────────────────────────────────────────────
# These are module-level constants — easy to find and tune.
# They must sum to 1.0.
TFIDF_WEIGHT: float = 0.40   # 40% weight on semantic similarity
SKILL_WEIGHT: float = 0.60   # 60% weight on skill keyword match rate


# ── Result Data Structure ─────────────────────────────────────────────────────
# WHY A DATACLASS?
#   We could return a plain dict, but a dataclass gives us:
#     - Type hints on every field (self-documenting)
#     - Auto-generated __repr__ for easy debugging
#     - IDE autocomplete when accessing fields
#     - Easy to convert to dict with dataclasses.asdict()
#   In Step 6, the Pydantic schema will mirror this structure exactly.

@dataclass
class AnalysisResult:
    """
    The complete output of one resume analysis run.
    Every field here becomes a key in the final JSON API response.
    """
    # ── Scores ────────────────────────────────────────────────────────────────
    tfidf_score:   float   # Raw TF-IDF cosine similarity (0.0 – 1.0)
    skill_score:   float   # Raw skill match rate (0.0 – 1.0)
    hybrid_score:  float   # Weighted combination of both (0.0 – 1.0)
    fit_percentage: float  # hybrid_score × 100, rounded to 1 decimal

    # ── Verdict ───────────────────────────────────────────────────────────────
    verdict: str           # "Strong Fit" | "Partial Fit" | "Weak Fit"
    advice:  str           # Human-readable recommendation

    # ── Skills ───────────────────────────────────────────────────────────────
    matched_skills: list[str]   # Skills in BOTH resume and JD
    missing_skills: list[str]   # Skills in JD but NOT in resume
    extra_skills:   list[str]   # Skills in resume but NOT in JD

    # ── Metadata ──────────────────────────────────────────────────────────────
    resume_skill_count: int     # Total skills found in resume
    jd_skill_count:     int     # Total skills found in JD
    resume_text_length: int     # Character count of extracted resume text

    # ── Optional debug info (not sent to API by default) ─────────────────────
    resume_text_preview: Optional[str] = field(default=None, repr=False)


# ── Hybrid Score Calculator ───────────────────────────────────────────────────

def compute_hybrid_score(tfidf_score: float, skill_match_rate: float) -> float:
    """
    Combine TF-IDF similarity and skill match rate into one score.

    This is a weighted average — simple, transparent, and explainable.
    "Explainability" matters in hiring tools: you must be able to justify
    why a candidate got a certain score. A weighted average is easy to explain.
    A neural network score is not.

    Args:
        tfidf_score:      Float 0.0–1.0 from TF-IDF cosine similarity.
        skill_match_rate: Float 0.0–1.0 fraction of JD skills found in resume.

    Returns:
        Weighted hybrid score, clamped to [0.0, 1.0], rounded to 4 decimals.
    """
    raw = (tfidf_score * TFIDF_WEIGHT) + (skill_match_rate * SKILL_WEIGHT)
    return round(max(0.0, min(1.0, raw)), 4)


# ── Main Pipeline Function ────────────────────────────────────────────────────

def analyze_resume(
    pdf_bytes: bytes,
    job_description: str,
) -> AnalysisResult:
    """
    Run the full resume analysis pipeline.

    This is the single public function of this module. The router (Step 6)
    calls only this function — it never touches the individual service modules.

    Args:
        pdf_bytes:       Raw bytes of the uploaded PDF resume.
        job_description: Plain text of the job description (from form field).

    Returns:
        AnalysisResult dataclass with all scores, skills, and verdict.

    Raises:
        PDFParsingError: If the uploaded file is not a valid PDF.
        EmptyPDFError:   If the PDF contains no extractable text.
        ValueError:      If the job description is empty.
    """
    logger.info("Starting resume analysis pipeline.")

    # ── Guard: validate job description ──────────────────────────────────────
    if not job_description or not job_description.strip():
        raise ValueError("Job description cannot be empty.")

    # ── Stage 1: Validate & parse PDF ────────────────────────────────────────
    logger.info("Stage 1: Validating and parsing PDF.")
    validate_pdf_bytes(pdf_bytes)              # Raises PDFParsingError if invalid
    resume_text = extract_text_from_pdf(pdf_bytes)  # Raises EmptyPDFError if blank

    # ── Stage 2: Extract skills ───────────────────────────────────────────────
    logger.info("Stage 2: Extracting skills.")
    resume_skills = extract_skills(resume_text)
    jd_skills     = extract_skills(job_description)

    logger.info(
        f"Skills found — Resume: {len(resume_skills)}, JD: {len(jd_skills)}"
    )

    # ── Stage 3: Compare skills ───────────────────────────────────────────────
    logger.info("Stage 3: Comparing skills.")
    skill_comparison = compare_skills(resume_skills, jd_skills)
    skill_match_rate = skill_comparison["match_rate"]  # 0.0 – 1.0

    # ── Stage 4: Compute TF-IDF similarity ───────────────────────────────────
    logger.info("Stage 4: Computing TF-IDF similarity score.")
    try:
        tfidf_score = compute_similarity_score(resume_text, job_description)
    except ValueError as e:
        # If TF-IDF fails (e.g. vocabulary too small), default to 0.0
        # and let the skill score carry the verdict.
        logger.warning(f"TF-IDF scoring failed, defaulting to 0.0. Reason: {e}")
        tfidf_score = 0.0

    # ── Stage 5: Compute hybrid score ─────────────────────────────────────────
    logger.info("Stage 5: Computing hybrid score.")
    hybrid_score   = compute_hybrid_score(tfidf_score, skill_match_rate)
    fit_percentage = round(hybrid_score * 100, 1)

    # ── Stage 6: Generate verdict ─────────────────────────────────────────────
    logger.info("Stage 6: Generating verdict.")
    breakdown = get_score_breakdown(hybrid_score)
    verdict   = breakdown["verdict"]
    advice    = breakdown["advice"]

    logger.info(
        f"Analysis complete — Hybrid: {hybrid_score}, Verdict: {verdict}"
    )

    # ── Stage 7: Assemble result ──────────────────────────────────────────────
    return AnalysisResult(
        # Scores
        tfidf_score    = tfidf_score,
        skill_score    = round(skill_match_rate, 4),
        hybrid_score   = hybrid_score,
        fit_percentage = fit_percentage,

        # Verdict
        verdict = verdict,
        advice  = advice,

        # Skills
        matched_skills = skill_comparison["matched_skills"],
        missing_skills = skill_comparison["missing_skills"],
        extra_skills   = skill_comparison["extra_skills"],

        # Metadata
        resume_skill_count = len(resume_skills),
        jd_skill_count     = len(jd_skills),
        resume_text_length = len(resume_text),

        # Debug preview (first 300 chars of resume text)
        resume_text_preview = resume_text[:300] if resume_text else None,
    )


# ── Convenience: Result to Dict ───────────────────────────────────────────────

def result_to_dict(result: AnalysisResult, include_debug: bool = False) -> dict:
    """
    Convert an AnalysisResult to a plain dict for JSON serialization.

    WHY NOT USE dataclasses.asdict()?
      dataclasses.asdict() is recursive and converts nested objects too —
      fine here, but we also want to control which fields are exposed.
      The debug field (resume_text_preview) is hidden by default.

    Args:
        result:        The AnalysisResult dataclass instance.
        include_debug: Whether to include the resume_text_preview field.

    Returns:
        Clean dict ready for JSON serialization.
    """
    output = {
        # Scores
        "tfidf_score":    result.tfidf_score,
        "skill_score":    result.skill_score,
        "fit_score":      result.hybrid_score,      # renamed for API clarity
        "fit_percentage": result.fit_percentage,

        # Verdict
        "verdict": result.verdict,
        "advice":  result.advice,

        # Skills
        "matched_skills": result.matched_skills,
        "missing_skills": result.missing_skills,
        "extra_skills":   result.extra_skills,

        # Metadata
        "resume_skill_count": result.resume_skill_count,
        "jd_skill_count":     result.jd_skill_count,
        "resume_text_length": result.resume_text_length,
    }

    if include_debug:
        output["resume_text_preview"] = result.resume_text_preview

    return output
