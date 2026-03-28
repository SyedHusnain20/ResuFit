"""
main.py — FastAPI Application Entry Point
─────────────────────────────────────────
Serves both the REST API and the static HTML frontend.
"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.config import settings
from app.exceptions import (
    configure_logging,
    http_exception_handler,
    unhandled_exception_handler,
    validation_exception_handler,
)
from app.middleware.logging import RequestLoggingMiddleware

configure_logging()

STATIC_DIR = Path(__file__).parent / "static"

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
            "Health check and root endpoints. `/health` is used by Render "
            "for uptime monitoring."
        ),
    },
]

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=settings.APP_DESCRIPTION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=OPENAPI_TAGS,
    contact={
        "name": "ResuFit on GitHub",
        "url": "https://github.com/YOUR_USERNAME/resufit",
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    },
)

# ── Middleware ────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RequestLoggingMiddleware)

# ── Exception Handlers ────────────────────────────────────────────────
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(StarletteHTTPException, http_exception_handler)
app.add_exception_handler(Exception, unhandled_exception_handler)

# ── API Routers ───────────────────────────────────────────────────────
from app.routers import analyze
app.include_router(analyze.router, prefix="/api/v1")

# ── Static Files ──────────────────────────────────────────────────────
# Serves CSS, JS, images from /app/static/ at the /static URL path.
# Must be mounted AFTER API routes so /api/v1/* routes take priority.
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ── Frontend Route ────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
def frontend():
    """
    Serve the ResuFit web app.
    include_in_schema=False hides this from the Swagger UI —
    it's a page, not an API endpoint.
    """
    return FileResponse(STATIC_DIR / "index.html")


# ── General API Routes ────────────────────────────────────────────────

@app.get("/health", tags=["General"], summary="Health check")
def health_check():
    """Returns HTTP 200 when the service is healthy. Used by Render."""
    return {"status": "ok"}


@app.get("/api", tags=["General"], summary="API info")
def api_info():
    """Returns API metadata and links."""
    return {
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running",
        "docs": "/docs",
        "analyze": "/api/v1/analyze",
    }
