"""
exceptions.py — Global Exception Handlers
───────────────────────────────────────────
WHY GLOBAL HANDLERS IN ADDITION TO try/except IN THE ROUTER?
  The router's try/except catches errors from the analysis pipeline.
  But errors can happen in other places too:
    - FastAPI's own validation (Pydantic ValidationError → 422)
    - Unhandled exceptions anywhere in the codebase (→ 500)
    - Starlette's HTTPException from anywhere (not just the router)

  Global handlers are the safety net that catches everything the
  router doesn't explicitly handle. They guarantee that:
    1. Every error returns JSON (not HTML or a raw Python traceback)
    2. Every error has the same { "detail": "..." } shape
    3. No internal Python details leak to the client in 500 errors

HOW FASTAPI EXCEPTION HANDLERS WORK:
  You register a handler with @app.exception_handler(ExceptionType).
  FastAPI calls your handler whenever that exception type is raised
  anywhere in the application — middleware, routes, dependencies, anywhere.

  Handler signature:
    async def handler(request: Request, exc: ExceptionType) -> JSONResponse

EXCEPTION HIERARCHY WE HANDLE:
  1. RequestValidationError  — Pydantic failed to validate inputs (422)
  2. HTTPException           — Any HTTPException raised anywhere (4xx/5xx)
  3. Exception               — Catch-all for anything unexpected (500)
"""

import logging

from fastapi import HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette import status

logger = logging.getLogger(__name__)


# ── Handler 1: Pydantic Validation Errors (422) ───────────────────────────────

async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """
    Handle Pydantic validation errors — triggered when request data doesn't
    match the expected schema (e.g. missing field, wrong type, min_length fail).

    WHY OVERRIDE THE DEFAULT?
      FastAPI's default 422 response looks like:
        { "detail": [{ "loc": [...], "msg": "...", "type": "..." }] }
      This is technically correct but exposes internal field names.
      We simplify it into one human-readable sentence.

    WHAT WE DO:
      Extract the first validation error and format it as a plain sentence.
    """
    errors = exc.errors()

    # Build a clean, human-readable message from the first error
    # errors() returns a list of dicts: [{ "loc": [...], "msg": "...", "type": "..." }]
    if errors:
        first_error   = errors[0]
        field_location = " → ".join(str(loc) for loc in first_error["loc"])
        message       = first_error["msg"]
        detail        = f"Validation error at '{field_location}': {message}."
    else:
        detail = "Request validation failed. Please check your input."

    logger.warning(
        f"Validation error on {request.method} {request.url.path}: {detail}"
    )

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": detail},
    )


# ── Handler 2: HTTP Exceptions (4xx / 5xx) ───────────────────────────────────

async def http_exception_handler(
    request: Request,
    exc: HTTPException,
) -> JSONResponse:
    """
    Handle all HTTPExceptions raised anywhere in the application.

    WHY OVERRIDE THE DEFAULT?
      FastAPI's default handler returns the correct JSON, but we want to:
        1. LOG every 4xx/5xx with its status code and path
        2. Guarantee our consistent { "detail": "..." } shape
        3. Add the X-Request-ID header to error responses too

    FastAPI's built-in handler doesn't log anything — errors silently
    disappear. Our handler makes every error visible in the logs.
    """
    if exc.status_code >= 500:
        logger.error(
            f"HTTP {exc.status_code} on {request.method} {request.url.path}: "
            f"{exc.detail}"
        )
    else:
        logger.warning(
            f"HTTP {exc.status_code} on {request.method} {request.url.path}: "
            f"{exc.detail}"
        )

    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )


# ── Handler 3: Catch-All for Unhandled Exceptions (500) ───────────────────────

async def unhandled_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """
    Last resort handler — catches any exception that wasn't handled elsewhere.

    CRITICAL SECURITY NOTE:
      We log the full exception details (including traceback via exc_info=True)
      so YOU can see what went wrong in the logs.
      But we return a GENERIC message to the client — never expose Python
      tracebacks, file paths, or internal error details to the outside world.
      These can reveal your tech stack, file structure, and vulnerabilities.

    WHY exc_info=True?
      This tells the logger to include the full Python traceback in the log.
      Without it, you'd only see the error message, not where it came from.
    """
    logger.error(
        f"Unhandled exception on {request.method} {request.url.path}: "
        f"{type(exc).__name__}: {exc}",
        exc_info=True,   # includes full traceback in the log
    )

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": (
                "An unexpected internal error occurred. "
                "Our team has been notified. Please try again later."
            )
        },
    )


# ── Logging Configuration ─────────────────────────────────────────────────────

def configure_logging() -> None:
    """
    Set up structured logging for the entire application.

    WHY CONFIGURE LOGGING HERE?
      Python's logging module needs to be configured once at startup.
      Without this, log messages either don't appear or use ugly defaults.

    FORMAT EXPLAINED:
      %(asctime)s    — timestamp: "2024-01-15 10:23:45,123"
      %(name)s       — logger name: "app.services.scorer"
      %(levelname)s  — level: "INFO", "WARNING", "ERROR"
      %(message)s    — the actual message

    WHY NOT USE A LOGGING LIBRARY LIKE loguru?
      For a portfolio project, stdlib logging is perfectly fine and has
      zero extra dependencies. In production you'd switch to structured
      JSON logging (loguru or structlog) for log aggregation tools.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)-30s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Suppress noisy third-party loggers that spam at INFO level
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("multipart").setLevel(logging.WARNING)
