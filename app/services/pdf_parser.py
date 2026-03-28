"""
pdf_parser.py — PDF Text Extraction Service
────────────────────────────────────────────
RESPONSIBILITY:
  Take raw PDF bytes (straight from the HTTP upload) and return a single
  clean string of text. Nothing more. This module does NOT score, extract
  skills, or make any decisions — it only extracts text.

WHY A SEPARATE MODULE?
  "Separation of concerns" — each file does exactly one job. If we later swap
  pdfplumber for PyMuPDF, we only change this file. The scorer, extractor, and
  router stay untouched.

WHY BYTES, NOT A FILE PATH?
  FastAPI gives us an UploadFile object whose content lives in memory as bytes.
  We never save the PDF to disk — that would slow things down and create
  security/cleanup headaches. pdfplumber can open a PDF directly from a
  BytesIO buffer, so we use that.

HOW pdfplumber WORKS:
  pdfplumber parses the internal PDF structure and extracts text with layout
  awareness (understands columns, headers, etc.). It returns text per-page,
  so we loop through all pages and join them into one string.
"""

import io
import logging
import re

import pdfplumber

# Module-level logger — best practice over bare print() in production code.
# Logs show up in your terminal during dev and in Render's log dashboard.
logger = logging.getLogger(__name__)


# ── Custom Exceptions ─────────────────────────────────────────────────────────
# WHY CUSTOM EXCEPTIONS?
#   Using a specific exception type (instead of generic Exception) lets the
#   router catch only PDF errors and return a clean 400 response, without
#   accidentally swallowing unrelated bugs.

class PDFParsingError(Exception):
    """Raised when a PDF cannot be opened or parsed."""
    pass


class EmptyPDFError(Exception):
    """Raised when a PDF opens successfully but contains no extractable text."""
    pass


# ── Main Extraction Function ──────────────────────────────────────────────────

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """
    Extract and clean all text from a PDF given its raw bytes.

    Args:
        file_bytes: Raw bytes of the uploaded PDF file.

    Returns:
        A single cleaned string containing all text from the PDF.

    Raises:
        PDFParsingError: If the bytes are not a valid PDF or can't be opened.
        EmptyPDFError:   If the PDF opens but yields no text (e.g. scanned image).
    """
    # ── Step 1: Wrap bytes in a file-like object ──────────────────────────────
    # io.BytesIO turns raw bytes into something that behaves like an open file.
    # pdfplumber.open() accepts either a file path (str) or a file-like object.
    # We use BytesIO so we never touch the disk.
    pdf_buffer = io.BytesIO(file_bytes)

    try:
        # ── Step 2: Open and iterate through pages ────────────────────────────
        # pdfplumber.open() returns a context manager — the `with` block ensures
        # the PDF is properly closed even if an error occurs mid-extraction.
        with pdfplumber.open(pdf_buffer) as pdf:

            logger.info(f"Opened PDF — {len(pdf.pages)} page(s) found.")

            page_texts = []

            for page_number, page in enumerate(pdf.pages, start=1):
                # extract_text() returns a string or None if the page is empty.
                # The `x_tolerance` and `y_tolerance` params control how
                # aggressively pdfplumber clusters nearby characters into words.
                # These defaults work well for standard resume layouts.
                raw_text = page.extract_text(x_tolerance=3, y_tolerance=3)

                if raw_text:
                    page_texts.append(raw_text)
                    logger.debug(
                        f"Page {page_number}: extracted {len(raw_text)} characters."
                    )
                else:
                    logger.warning(
                        f"Page {page_number}: no text found (possibly an image)."
                    )

    except Exception as e:
        # Catches corrupt PDFs, password-protected PDFs, or non-PDF bytes.
        logger.error(f"Failed to parse PDF: {e}")
        raise PDFParsingError(
            f"Could not read the PDF file. Make sure it is a valid, "
            f"non-password-protected PDF. Details: {str(e)}"
        )

    # ── Step 3: Guard against empty result ────────────────────────────────────
    if not page_texts:
        raise EmptyPDFError(
            "The PDF opened successfully but contained no extractable text. "
            "It may be a scanned image-based PDF. Please upload a text-based PDF."
        )

    # ── Step 4: Join pages and clean the text ─────────────────────────────────
    # Pages are joined with double newline to preserve section breaks.
    combined_text = "\n\n".join(page_texts)
    cleaned_text  = _clean_text(combined_text)

    logger.info(
        f"Extraction complete — {len(cleaned_text)} characters after cleaning."
    )
    return cleaned_text


# ── Text Cleaning Helper ──────────────────────────────────────────────────────

def _clean_text(text: str) -> str:
    """
    Normalize raw extracted text for downstream NLP and TF-IDF processing.

    WHY CLEAN?
      Raw PDF text often contains:
        - Multiple consecutive blank lines (layout artifacts)
        - Runs of spaces from column alignment
        - Bullet point symbols (•, ▪, ●) that confuse tokenizers
        - Non-breaking spaces (\\xa0) from PDF encodings

    This function standardizes all of that into clean, consistent whitespace.

    Args:
        text: Raw text string from pdfplumber.

    Returns:
        Cleaned text string.
    """
    # Replace non-breaking spaces and other unicode whitespace with regular space
    text = text.replace("\xa0", " ").replace("\u200b", "")

    # Replace common bullet symbols with a space so words don't merge
    # e.g. "Python•FastAPI" → "Python FastAPI"
    text = re.sub(r"[•▪●■◦▸→✓✔-]", " ", text)

    # Collapse runs of spaces (but NOT newlines) into a single space
    text = re.sub(r"[ \t]+", " ", text)

    # Collapse more than 2 consecutive newlines into exactly 2
    # (preserves section breaks without huge gaps)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Strip leading/trailing whitespace from each line
    lines = [line.strip() for line in text.splitlines()]

    # Remove lines that are entirely empty after stripping
    # (keeps single blank lines that separate sections)
    cleaned_lines = []
    prev_blank = False
    for line in lines:
        is_blank = len(line) == 0
        if is_blank and prev_blank:
            continue  # skip consecutive blank lines
        cleaned_lines.append(line)
        prev_blank = is_blank

    return "\n".join(cleaned_lines).strip()


# ── Utility: Basic Validation ─────────────────────────────────────────────────

def validate_pdf_bytes(file_bytes: bytes) -> None:
    """
    Quick sanity check before attempting full parsing.

    WHY THIS EXISTS:
      pdfplumber gives a cryptic error if you hand it a .docx or a JPEG.
      This check catches those cases early and gives a clear error message.
      PDFs always start with the magic bytes: %PDF

    Args:
        file_bytes: Raw bytes to check.

    Raises:
        PDFParsingError: If the bytes don't look like a PDF.
    """
    # PDF files always begin with the ASCII sequence "%PDF"
    if not file_bytes.startswith(b"%PDF"):
        raise PDFParsingError(
            "The uploaded file does not appear to be a valid PDF. "
            "Only PDF files are supported."
        )

    if len(file_bytes) < 100:
        raise PDFParsingError(
            "The uploaded file is too small to be a valid PDF."
        )
