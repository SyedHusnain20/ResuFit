"""
schemas/analysis.py — Pydantic Request & Response Models
──────────────────────────────────────────────────────────
WHY PYDANTIC SCHEMAS?
  FastAPI uses Pydantic models to:
    1. VALIDATE inputs automatically — wrong types get rejected with a clear
       error before your code even runs
    2. SERIALIZE outputs automatically — your Python objects become JSON
    3. DOCUMENT the API — Swagger UI reads these models and shows exactly
       what fields are expected and returned, with types and descriptions
    4. PROVIDE editor support — full autocomplete on request/response fields

HOW FASTAPI + PYDANTIC WORK TOGETHER:
  - You define a Pydantic model (a class inheriting from BaseModel)
  - You use it as a type hint in your route function
  - FastAPI handles validation, parsing, and serialization automatically
  - If validation fails, FastAPI returns HTTP 422 with a detailed error message

PYDANTIC V2 NOTE:
  We're using Pydantic v2 (required by FastAPI 0.111+).
  Key differences from v1:
    - `model_config` replaces `class Config`
    - `model_json_schema()` replaces `schema()`
    - Validation is significantly faster (Rust-based core)
"""

from pydantic import BaseModel, Field


# ── Response Model ────────────────────────────────────────────────────────────
# This defines EXACTLY what the API returns for every successful analysis.
# Every field has:
#   - A type annotation   (for validation and serialization)
#   - A Field() with description (shows up in Swagger UI docs)
#   - An example value   (makes the Swagger UI "Try it out" feature useful)

class AnalysisResponse(BaseModel):
    """
    The complete response returned by POST /api/v1/analyze.
    All fields are always present — no optional fields in the response.
    """

    # ── Scores ────────────────────────────────────────────────────────────────
    fit_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Hybrid fit score combining TF-IDF similarity (40%) and skill "
            "match rate (60%). Range: 0.0 (no match) to 1.0 (perfect match)."
        ),
        examples=[0.58],
    )

    fit_percentage: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="fit_score expressed as a percentage (0.0 – 100.0).",
        examples=[58.2],
    )

    tfidf_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Raw TF-IDF cosine similarity between resume and job description. "
            "Captures semantic and contextual similarity."
        ),
        examples=[0.35],
    )

    skill_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Fraction of job description skills found in the resume. "
            "Range: 0.0 (none matched) to 1.0 (all matched)."
        ),
        examples=[0.78],
    )

    # ── Verdict ───────────────────────────────────────────────────────────────
    verdict: str = Field(
        ...,
        description="Human-readable hiring verdict based on the fit score.",
        examples=["Partial Fit"],
    )

    advice: str = Field(
        ...,
        description="Actionable recommendation for the recruiter or candidate.",
        examples=["This candidate partially matches the job requirements."],
    )

    # ── Skills ───────────────────────────────────────────────────────────────
    matched_skills: list[str] = Field(
        ...,
        description="Skills present in BOTH the resume and job description.",
        examples=[["python", "fastapi", "docker", "aws"]],
    )

    missing_skills: list[str] = Field(
        ...,
        description=(
            "Skills required by the job description that are absent "
            "from the resume. These are the candidate's skill gaps."
        ),
        examples=[["kubernetes", "terraform"]],
    )

    extra_skills: list[str] = Field(
        ...,
        description=(
            "Skills on the resume not mentioned in the job description. "
            "May indicate additional value the candidate brings."
        ),
        examples=[["redis", "celery", "pytest"]],
    )

    # ── Metadata ──────────────────────────────────────────────────────────────
    resume_skill_count: int = Field(
        ...,
        ge=0,
        description="Total number of skills detected in the resume.",
        examples=[14],
    )

    jd_skill_count: int = Field(
        ...,
        ge=0,
        description="Total number of skills detected in the job description.",
        examples=[9],
    )

    resume_text_length: int = Field(
        ...,
        ge=0,
        description="Character count of text extracted from the resume PDF.",
        examples=[2400],
    )

    # ── Pydantic v2 config ────────────────────────────────────────────────────
    # `from_attributes=True` allows creating this model from a dataclass or
    # ORM object using model_validate(obj) — not needed now but good practice.
    model_config = {"from_attributes": True}


# ── Error Response Model ──────────────────────────────────────────────────────
# WHY A SEPARATE ERROR MODEL?
#   Consistent error responses are a mark of a well-designed API.
#   Instead of returning raw FastAPI error dicts, we wrap errors in a
#   predictable structure. API consumers can always expect { "detail": "..." }.

class ErrorResponse(BaseModel):
    """
    Standard error response returned for all 4xx errors.
    FastAPI also uses this shape for its built-in validation errors (422).
    """
    detail: str = Field(
        ...,
        description="Human-readable description of what went wrong.",
        examples=["The uploaded file does not appear to be a valid PDF."],
    )
