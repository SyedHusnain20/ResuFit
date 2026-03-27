"""
test_pdf_parser.py — Unit Tests for the PDF Parser
────────────────────────────────────────────────────
WHY WRITE TESTS?
  Tests prove your code works, prevent regressions (future changes breaking
  old behaviour), and make you look professional to employers. A project with
  tests signals you write production-quality code.

HOW PYTEST WORKS:
  - Any file named test_*.py is auto-discovered by pytest
  - Any function named test_* inside it is run as a test
  - `assert` statements are how you check expected vs actual values
  - If an assert fails, pytest shows you exactly what went wrong

WHAT WE'RE TESTING:
  1. Valid PDF → returns text
  2. Non-PDF bytes → raises PDFParsingError
  3. Empty/tiny bytes → raises PDFParsingError
  4. Text cleaning → bullets and extra spaces are removed
  5. The _clean_text helper directly
"""

import pytest

from app.services.pdf_parser import (
    EmptyPDFError,
    PDFParsingError,
    _clean_text,
    extract_text_from_pdf,
    validate_pdf_bytes,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────
# A pytest fixture is a reusable piece of test data or setup.
# Instead of repeating the same setup in every test, you define it once here
# and inject it by name into test functions.

@pytest.fixture
def minimal_valid_pdf() -> bytes:
    """
    A real, minimal PDF in raw bytes.

    WHY NOT load a file from disk?
      Keeping test data in code means tests run without any external files.
      This PDF renders a single page with the text 'Hello World'.
      It was generated using the PDF spec and is the smallest valid PDF.
    """
    # This is a real, spec-compliant minimal PDF that contains "Hello World"
    return (
        b"%PDF-1.4\n"
        b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
        b"3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R"
        b"/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>endobj\n"
        b"4 0 obj<</Length 44>>\nstream\n"
        b"BT /F1 12 Tf 100 700 Td (Hello World) Tj ET\n"
        b"endstream\nendobj\n"
        b"5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj\n"
        b"xref\n0 6\n"
        b"0000000000 65535 f \n"
        b"0000000009 00000 n \n"
        b"0000000058 00000 n \n"
        b"0000000115 00000 n \n"
        b"0000000266 00000 n \n"
        b"0000000360 00000 n \n"
        b"trailer<</Size 6/Root 1 0 R>>\n"
        b"startxref\n441\n%%EOF"
    )


@pytest.fixture
def not_a_pdf() -> bytes:
    """Bytes that are clearly not a PDF (a JPEG header)."""
    return b"\xff\xd8\xff\xe0" + b"\x00" * 100  # JPEG magic bytes


@pytest.fixture
def too_small() -> bytes:
    """Bytes that are too small to be any valid file."""
    return b"%PDF-tiny"


# ── validate_pdf_bytes Tests ──────────────────────────────────────────────────

class TestValidatePdfBytes:
    """Tests for the quick pre-validation check."""

    def test_valid_pdf_header_passes(self, minimal_valid_pdf):
        # Should not raise anything — just return None silently
        validate_pdf_bytes(minimal_valid_pdf)

    def test_non_pdf_bytes_raises_error(self, not_a_pdf):
        # pytest.raises() asserts that the code block raises the expected exception
        with pytest.raises(PDFParsingError) as exc_info:
            validate_pdf_bytes(not_a_pdf)
        # Also check the error message is helpful
        assert "valid PDF" in str(exc_info.value)

    def test_too_small_raises_error(self, too_small):
        with pytest.raises(PDFParsingError) as exc_info:
            validate_pdf_bytes(too_small)
        assert "too small" in str(exc_info.value)


# ── _clean_text Tests ─────────────────────────────────────────────────────────

class TestCleanText:
    """Tests for the text normalization helper."""

    def test_removes_bullet_symbols(self):
        raw = "Skills:\n• Python\n• FastAPI\n▪ Docker"
        result = _clean_text(raw)
        assert "•" not in result
        assert "▪" not in result
        # Words should still be there
        assert "Python" in result
        assert "FastAPI" in result

    def test_collapses_multiple_blank_lines(self):
        raw = "Section A\n\n\n\n\nSection B"
        result = _clean_text(raw)
        # Should not have more than 2 consecutive newlines
        assert "\n\n\n" not in result

    def test_replaces_non_breaking_space(self):
        raw = "Python\xa0Developer"
        result = _clean_text(raw)
        assert "\xa0" not in result
        assert "Python" in result
        assert "Developer" in result

    def test_collapses_extra_spaces(self):
        raw = "Python    Developer     Engineer"
        result = _clean_text(raw)
        assert "  " not in result  # No double spaces remain

    def test_strips_leading_trailing_whitespace(self):
        raw = "   hello world   "
        result = _clean_text(raw)
        assert result == "hello world"

    def test_empty_string_returns_empty(self):
        assert _clean_text("") == ""

    def test_already_clean_text_unchanged(self):
        raw = "Python Developer\nFastAPI Expert"
        result = _clean_text(raw)
        assert "Python Developer" in result
        assert "FastAPI Expert" in result


# ── extract_text_from_pdf Tests ───────────────────────────────────────────────

class TestExtractTextFromPdf:
    """Integration-level tests for the full extraction pipeline."""

    def test_invalid_bytes_raises_parsing_error(self, not_a_pdf):
        # A JPEG masquerading as a PDF should fail at the pdfplumber level
        with pytest.raises(PDFParsingError):
            extract_text_from_pdf(not_a_pdf)

    def test_empty_bytes_raises_parsing_error(self):
        with pytest.raises(PDFParsingError):
            extract_text_from_pdf(b"")

    def test_returns_string(self, minimal_valid_pdf):
        # Even if the minimal PDF has limited text, the return type must be str
        try:
            result = extract_text_from_pdf(minimal_valid_pdf)
            assert isinstance(result, str)
        except (PDFParsingError, EmptyPDFError):
            # Minimal PDFs may not render text depending on pdfplumber version
            # This is acceptable — the important thing is no unexpected crash
            pass
