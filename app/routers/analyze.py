"""
routers/analyze.py — Resume Analysis Endpoint
───────────────────────────────────────────────
RESPONSIBILITY:
  Handle HTTP concerns ONLY:
    - Receive the file upload and form field
    - Validate file size and content type
    - Call the analyzer pipeline
    - Map service exceptions to appropriate HTTP responses
    - Return the structured JSON response

  The router does NOT contain any business logic. It delegates everything
  to analyzer.py. This is the "thin controller" pattern — routers are just
  traffic cops that direct requests to the right service.

WHY AN APIROUTER INSTEAD OF PUTTING ROUTES IN main.py?
  APIRouter lets you group related routes and mount them with a prefix.
  All routes in this file get the prefix "/api/v1" (set in main.py).
  This means:
    - Routes here write @router.post("/analyze") not @app.post("/api/v1/analyze")
    - You can version your API by mounting v2 routes without touching v1
    - main.py stays clean — it just mounts routers, not individual routes

HOW FASTAPI FILE UPLOADS WORK:
  FastAPI uses `UploadFile` for file uploads and `Form` for form text fields.
  Both come from `fastapi` directly. When the client sends a multipart/form-data
  request (which is what HTML forms and most HTTP clients use for file uploads),
  FastAPI automatically parses it and injects the values.

  WHY NOT JSON for file uploads?
    JSON can't natively carry binary data. For binary files, multipart/form-data
    is the standard. The client sends the PDF as a file part and the job
    description as a text part in the same request.
"""

import logging

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from app.config import settings
from app.schemas.analysis import AnalysisResponse
from app.services.analyzer import analyze_resume, result_to_dict
from app.services.pdf_parser import EmptyPDFError, PDFParsingError

logger = logging.getLogger(__name__)

# ── Router instance ───────────────────────────────────────────────────────────
# The prefix "/api/v1" is added in main.py when this router is included.
# The tag "Analysis" groups this endpoint under its own section in Swagger UI.
router = APIRouter(tags=["Analysis"])


# ── POST /api/v1/analyze ──────────────────────────────────────────────────────

@router.post(
    "/analyze",
    response_model=AnalysisResponse,
    status_code=status.HTTP_200_OK,
    summary="Analyze a resume against a job description",
    description="""
Upload a resume PDF and provide a job description to receive a complete
fit analysis including:

- **fit_score**: Hybrid score (0.0–1.0) combining TF-IDF and skill matching
- **verdict**: Strong Fit / Partial Fit / Weak Fit
- **matched_skills**: Skills present in both resume and JD
- **missing_skills**: Skills required by JD but absent from resume
- **extra_skills**: Additional skills the candidate brings

The resume must be a **text-based PDF** (not a scanned image).
Maximum file size: **5 MB**.
    """,
    responses={
        200: {"description": "Analysis completed successfully."},
        400: {"description": "Invalid PDF file or empty job description."},
        413: {"description": "File exceeds the 5 MB size limit."},
        422: {"description": "Validation error — missing required fields."},
        500: {"description": "Unexpected server error."},
    },
)
async def analyze(
    resume: UploadFile = File(
        ...,
        description="Resume file in PDF format (max 5 MB, text-based only).",
    ),
    job_description: str = Form(
        ...,
        min_length=50,
        description=(
            "Full text of the job description. "
            "Minimum 50 characters for meaningful analysis."
        ),
    ),
) -> AnalysisResponse:
    """
    Main resume analysis endpoint.

    Accepts a multipart/form-data request with:
      - `resume`:          PDF file upload
      - `job_description`: Plain text form field (min 50 chars)

    Returns a full AnalysisResponse JSON object on success.
    """

    # ── Step 1: Validate content type ─────────────────────────────────────────
    # Check MIME type before reading file bytes — fail fast
    # content_type can be None if the client doesn't set it, so we handle that
    if resume.content_type not in settings.ALLOWED_CONTENT_TYPES:
        logger.warning(
            f"Rejected upload — content_type: {resume.content_type}"
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Invalid file type '{resume.content_type}'. "
                f"Only PDF files are accepted (application/pdf)."
            ),
        )

    # ── Step 2: Read file bytes ───────────────────────────────────────────────
    # await is required because UploadFile.read() is async —
    # it reads from the request stream without blocking other requests.
    file_bytes = await resume.read()

    # ── Step 3: Validate file size ────────────────────────────────────────────
    # We check size AFTER reading because the content-length header can be
    # spoofed. Reading the actual bytes and checking len() is the safe approach.
    if len(file_bytes) > settings.MAX_FILE_SIZE_BYTES:
        size_mb = len(file_bytes) / (1024 * 1024)
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                f"File size {size_mb:.1f} MB exceeds the 5 MB limit. "
                f"Please upload a smaller PDF."
            ),
        )

    if len(file_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    logger.info(
        f"Received upload — filename: '{resume.filename}', "
        f"size: {len(file_bytes)} bytes"
    )

    # ── Step 4: Run the analysis pipeline ────────────────────────────────────
    # We delegate ALL business logic to the analyzer service.
    # The router's job here is just to map service exceptions → HTTP errors.
    try:
        result = analyze_resume(
            pdf_bytes=file_bytes,
            job_description=job_description,
        )

    except PDFParsingError as e:
        # PDF is corrupt, password-protected, or not a PDF at all
        logger.warning(f"PDF parsing failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    except EmptyPDFError as e:
        # PDF opened but contained no extractable text (likely scanned image)
        logger.warning(f"Empty PDF: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    except ValueError as e:
        # Empty or invalid job description text
        logger.warning(f"Invalid input: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    except Exception as e:
        # Unexpected errors — log the full traceback, return generic 500
        # We deliberately don't expose internal error details to the client
        logger.error(f"Unexpected error during analysis: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "An unexpected error occurred during analysis. "
                "Please try again or contact support."
            ),
        )

    # ── Step 5: Serialize and return ──────────────────────────────────────────
    # result_to_dict() converts the AnalysisResult dataclass to a plain dict.
    # FastAPI then validates it against AnalysisResponse (our Pydantic model)
    # and serializes it to JSON automatically.
    output = result_to_dict(result)

    logger.info(
        f"Analysis complete — verdict: {result.verdict}, "
        f"score: {result.hybrid_score}"
    )

    # FastAPI validates this dict against AnalysisResponse before returning
    return AnalysisResponse(**output)
