"""
main.py — FastAPI Application Entry Point
─────────────────────────────────────────
Registers middleware, global exception handlers, routers, and OpenAPI metadata.
"""

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.config import settings
from app.exceptions import (
    configure_logging,
    http_exception_handler,
    unhandled_exception_handler,
    validation_exception_handler,
)
from app.middleware.logging import RequestLoggingMiddleware

# ── Configure logging first ───────────────────────────────────────────────────
configure_logging()

# ── OpenAPI Tag Descriptions ──────────────────────────────────────────────────
# These appear as section headers in the Swagger UI (/docs), making the API
# self-documenting. Each tag groups related endpoints with a description.
OPENAPI_TAGS = [
    {
        "name": "Analysis",
        "description": (
            "Core resume analysis endpoint. Upload a PDF resume and job description "
            "to receive a hybrid fit score, skill breakdown, and hiring verdict."
        ),
    },
    {
        "name": "General",
        "description": (
            "Health check and root endpoints. The `/health` endpoint is used by "
            "Render for uptime monitoring and auto-restart on failure."
        ),
    },
]

# ── App Instance ──────────────────────────────────────────────────────────────
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=settings.APP_DESCRIPTION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=OPENAPI_TAGS,
    # Shown in Swagger UI header — links to your GitHub repo
    contact={
        "name": "ResuFit on GitHub",
        "url": "https://github.com/YOUR_USERNAME/resufit",
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    },
)

# ── Middleware ────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RequestLoggingMiddleware)

# ── Global Exception Handlers ─────────────────────────────────────────────────
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(StarletteHTTPException, http_exception_handler)
app.add_exception_handler(Exception, unhandled_exception_handler)

# ── Routers ───────────────────────────────────────────────────────────────────
from app.routers import analyze
app.include_router(analyze.router, prefix="/api/v1")

# ── General Routes ────────────────────────────────────────────────────────────

@app.get(
    "/",
    tags=["General"],
    summary="API root — status and links",
)
def root():
    """
    Returns the API name, version, current status, and a link to the
    interactive documentation. Use this as a quick sanity check that
    the service is running.
    """
    return {
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running",
        "docs": "/docs",
        "health": "/health",
        "analyze": "/api/v1/analyze",
    }


@app.get(
    "/health",
    tags=["General"],
    summary="Health check for uptime monitoring",
)
def health_check():
    """
    Returns `{ "status": "ok" }` with HTTP 200 when the service is healthy.

    **Used by Render** to verify the service is alive after every deploy.
    If this endpoint doesn't return 200, Render will restart the service.
    Also useful for external uptime monitors (UptimeRobot, Pingdom, etc.).
    """
    return {"status": "ok"}
