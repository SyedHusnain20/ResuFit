"""
skill_extractor.py — Skill Extraction Service
──────────────────────────────────────────────
RESPONSIBILITY:
  Given a plain text string (from the PDF parser or raw job description input),
  return the set of skills found in that text by matching against our taxonomy.

THE CORE ALGORITHM — Boundary-Aware Keyword Matching:
  The naive approach is: `if skill in text`. That fails badly:
    - "r" would match inside "javascript" or "developer"
    - "c" would match everywhere
    - "go" would match "good", "google", "django"

  The correct approach is regex word-boundary matching.
  A word boundary matches between a word character and a non-word character.
  So the pattern for "go" matches it as a standalone word but NOT inside "good" or "django".

  For multi-word skills like "machine learning":
    - We use boundary assertions on the outer edges
    - Internal spaces are matched with whitespace patterns (one or more chars)
    - This handles "machine  learning" (double space) and newline-separated words

PERFORMANCE NOTE:
  We compile all regex patterns once at module load time (not per request).
  Compiling a regex is expensive — calling re.compile() on every API request
  would be wasteful. Stored in a module-level dict, compilation happens once.
"""

import logging
import re

from app.services.skills_taxonomy import (
    ALL_SKILLS,
    SKILL_TO_CATEGORY,
    SKILLS_TAXONOMY,
)

logger = logging.getLogger(__name__)


# ── Pre-compile Regex Patterns ────────────────────────────────────────────────
# WHY PRE-COMPILE?
#   re.compile() parses the regex string into an internal state machine.
#   Doing this once at startup (module load) instead of on every request
#   gives a meaningful speedup — especially for 500+ patterns.
#
# WHY re.IGNORECASE?
#   Resumes write "Python", "PYTHON", "python" interchangeably.
#   Case-insensitive matching handles all variants without lowercasing
#   the source text (which would destroy proper nouns and acronyms we
#   might want to preserve elsewhere).

def _build_patterns() -> dict[str, re.Pattern]:
    """
    Build a compiled regex pattern for every skill in the taxonomy.

    WHY NOT USE \\b FOR EVERYTHING?
      '\\b' is a word boundary — the transition between a word character (\\w)
      and a non-word character (\\W). It works perfectly for "python" or "go",
      but FAILS for skills like "c++" that END with a non-word character (+).
      After re.escape, "c++" becomes "c\\+\\+" — the final "+" is already a
      non-word char, so "\\b" after it doesn't match a word boundary correctly.

      The fix: use lookahead/lookbehind assertions instead:
        (?<!\\w)  = "not preceded by a word character"  (replaces leading \\b)
        (?!\\w)   = "not followed by a word character"  (replaces trailing \\b)
      These work correctly regardless of whether the skill starts or ends with
      a word character or a symbol.

    Returns:
        Dict mapping skill string -> compiled regex pattern.
    """
    patterns = {}
    for skill in ALL_SKILLS:
        # Escape special regex chars in skill names (e.g. "c++" -> "c\\+\\+")
        escaped = re.escape(skill)

        # Replace escaped spaces with \\s+ to match any whitespace between words
        # re.escape turns " " into "\\ " so we replace that with \\s+
        escaped = escaped.replace(r"\ ", r"\s+")

        # Use lookahead/lookbehind instead of \\b for universal boundary matching
        pattern = re.compile(rf"(?<!\w){escaped}(?!\w)", re.IGNORECASE)
        patterns[skill] = pattern

    logger.debug(f"Compiled {len(patterns)} skill patterns.")
    return patterns


# Module-level: compiled once when the module is first imported
_SKILL_PATTERNS: dict[str, re.Pattern] = _build_patterns()


# ── Main Extraction Function ──────────────────────────────────────────────────

def extract_skills(text: str) -> set[str]:
    """
    Find all skills from the taxonomy that appear in the given text.

    Uses pre-compiled word-boundary regex patterns for accurate matching.
    Returns skills in their canonical (lowercase) taxonomy form.

    Args:
        text: Plain text from a resume or job description.

    Returns:
        Set of matched skill strings (e.g. {"python", "fastapi", "docker"}).
        Returns empty set if no skills are found or text is empty.
    """
    if not text or not text.strip():
        logger.warning("extract_skills called with empty text.")
        return set()

    matched: set[str] = set()

    for skill, pattern in _SKILL_PATTERNS.items():
        if pattern.search(text):
            matched.add(skill)

    logger.info(f"Extracted {len(matched)} skills from text.")
    return matched


# ── Comparison Function ───────────────────────────────────────────────────────

def compare_skills(
    resume_skills: set[str],
    jd_skills: set[str],
) -> dict:
    """
    Compare resume skills against job description skills.

    WHY A SEPARATE FUNCTION?
      The extractor's job is extraction. Comparison is a different concern.
      Keeping them separate makes each easier to test independently.

    Args:
        resume_skills: Skills found in the resume.
        jd_skills:     Skills found in the job description.

    Returns:
        A dict with:
          - matched_skills:  Skills present in BOTH resume and JD
          - missing_skills:  Skills in JD but NOT in resume
          - extra_skills:    Skills in resume but NOT in JD (bonus info)
          - match_rate:      Fraction of JD skills covered (0.0 – 1.0)
    """
    # Set intersection: skills the candidate HAS that the JD REQUIRES
    matched = resume_skills & jd_skills

    # Set difference: skills the JD REQUIRES that the candidate is MISSING
    missing = jd_skills - resume_skills

    # Skills on the resume not mentioned in JD (useful context, not penalised)
    extra = resume_skills - jd_skills

    # Avoid division by zero if JD has no recognisable skills
    match_rate = len(matched) / len(jd_skills) if jd_skills else 0.0

    return {
        "matched_skills": sorted(matched),   # sorted for deterministic output
        "missing_skills": sorted(missing),
        "extra_skills":   sorted(extra),
        "match_rate":     round(match_rate, 4),
    }


# ── Category Breakdown ────────────────────────────────────────────────────────

def get_skills_by_category(skills: set[str]) -> dict[str, list[str]]:
    """
    Group a flat set of skills into their taxonomy categories.

    Useful for richer API responses like:
      { "Machine Learning & AI": ["pytorch", "scikit-learn"],
        "Cloud & DevOps": ["docker", "aws"] }

    Args:
        skills: A flat set of skill strings.

    Returns:
        Dict of { category: [skill, skill, ...] }, sorted alphabetically.
        Empty categories are omitted.
    """
    grouped: dict[str, list[str]] = {}

    for skill in skills:
        category = SKILL_TO_CATEGORY.get(skill, "Other")
        grouped.setdefault(category, []).append(skill)

    # Sort skills within each category for consistent output
    return {cat: sorted(skills_list) for cat, skills_list in sorted(grouped.items())}


# ── Utility: Skill Count Per Category ────────────────────────────────────────

def taxonomy_summary() -> dict[str, int]:
    """
    Return a count of skills per category — useful for debugging and docs.

    Example output:
      { "Programming Languages": 29, "Machine Learning & AI": 35, ... }
    """
    return {
        category: len(skills)
        for category, skills in SKILLS_TAXONOMY.items()
    }
