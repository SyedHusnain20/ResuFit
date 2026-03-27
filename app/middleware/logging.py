"""
middleware/logging.py — Request Logging Middleware
───────────────────────────────────────────────────
WHAT IS MIDDLEWARE?
  Middleware is code that runs on EVERY request before it reaches your route
  handler and on EVERY response before it's sent back to the client.
  Think of it as a wrapper around your entire application.

  Request flow with middleware:
    Client → Middleware (before) → Route Handler → Middleware (after) → Client

WHY LOG EVERY REQUEST?
  In production, when something breaks you need to know:
    - Which endpoint was called?
    - What was the response status code?
    - How long did it take?
    - Was there a pattern of errors?
  Without request logging, debugging production issues is nearly impossible.

WHAT WE LOG:
  - Method + URL path (what was called)
  - Response status code (did it succeed or fail?)
  - Processing time in milliseconds (performance monitoring)
  - Request ID (unique ID per request for tracing errors across log lines)

WHY A UNIQUE REQUEST ID?
  When multiple requests arrive simultaneously, their log lines interleave.
  A unique ID per request lets you filter all logs for one specific request.
  This is called "request tracing" and is standard in production systems.
  Real systems use distributed tracing tools (Jaeger, Datadog) — this is
  the simple, self-contained version.
"""

import logging
import time
import uuid

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Logs every incoming request and outgoing response with timing.

    Inherits from BaseHTTPMiddleware (from Starlette, which FastAPI is built on).
    You override dispatch() to wrap the request/response cycle.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        """
        Called for every request. `call_next` passes the request to the
        next handler in the chain (eventually reaching your route function).

        Args:
            request:   The incoming HTTP request object.
            call_next: Callable that passes the request to the next handler.

        Returns:
            The HTTP response object.
        """
        # ── Generate a unique request ID ──────────────────────────────────────
        # uuid4() generates a random UUID. We take the first 8 chars to keep
        # logs readable. Collision probability at this scale is negligible.
        request_id = str(uuid.uuid4())[:8]

        # ── Record start time ─────────────────────────────────────────────────
        # time.perf_counter() is the most precise timer in Python.
        # It's better than time.time() for measuring short durations.
        start_time = time.perf_counter()

        # ── Log the incoming request ──────────────────────────────────────────
        logger.info(
            f"[{request_id}] → {request.method} {request.url.path}"
        )

        # ── Process the request ────────────────────────────────────────────────
        # call_next() hands off to the route handler and waits for the response.
        # If the route handler raises an unhandled exception, it propagates here.
        try:
            response = await call_next(request)
        except Exception as e:
            # If something catastrophic happens before we get a response,
            # log it and re-raise so FastAPI's exception handlers can catch it.
            duration_ms = (time.perf_counter() - start_time) * 1000
            logger.error(
                f"[{request_id}] ✗ {request.method} {request.url.path} "
                f"— UNHANDLED ERROR after {duration_ms:.1f}ms: {e}"
            )
            raise

        # ── Calculate duration ────────────────────────────────────────────────
        duration_ms = (time.perf_counter() - start_time) * 1000

        # ── Log the response ──────────────────────────────────────────────────
        # Choose log level based on status code:
        #   2xx → INFO  (success)
        #   4xx → WARNING (client error — their fault)
        #   5xx → ERROR  (server error — our fault)
        status_code = response.status_code
        if status_code >= 500:
            log_fn = logger.error
        elif status_code >= 400:
            log_fn = logger.warning
        else:
            log_fn = logger.info

        log_fn(
            f"[{request_id}] ← {status_code} {request.method} "
            f"{request.url.path} — {duration_ms:.1f}ms"
        )

        # ── Attach request ID to response headers ─────────────────────────────
        # This lets API consumers trace their request in your logs.
        # X- prefix is the convention for custom HTTP headers.
        response.headers["X-Request-ID"] = request_id

        return response
