"""
scorer.py — TF-IDF Scoring Engine
───────────────────────────────────
RESPONSIBILITY:
  Take two plain text strings (resume + job description) and return a
  float between 0.0 and 1.0 representing how well they match semantically.

THE ALGORITHM IN PLAIN ENGLISH:
  Step 1 — Preprocess:  Clean and normalise both texts.
  Step 2 — Vectorize:   Convert both texts into TF-IDF vectors.
                        Each dimension = one unique word.
                        Each value    = how important that word is.
  Step 3 — Score:       Compute cosine similarity between the two vectors.
                        Result is a float from 0.0 (no match) to 1.0 (identical).

WHY TF-IDF OVER SIMPLE WORD OVERLAP?
  Word overlap counts shared words equally. TF-IDF weights them by rarity.
  In a resume context:
    - "the", "and", "with" appear everywhere → near-zero TF-IDF weight
    - "kubernetes", "transformer", "fastapi" are rare → high TF-IDF weight
  This means a match on "kubernetes" contributes far more to the score than
  a match on "experience" — which is exactly what we want.

WHY COSINE SIMILARITY OVER EUCLIDEAN DISTANCE?
  A 2-page resume and a 1-paragraph JD are very different in length.
  Euclidean distance would penalise length differences heavily.
  Cosine similarity only cares about direction (which words matter),
  not magnitude (how many words total). Perfect for document comparison.

SCIKIT-LEARN CLASSES USED:
  TfidfVectorizer: Converts text → sparse TF-IDF matrix.
  cosine_similarity: Computes dot product of normalised vectors.
"""

import logging
import re
import string

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.config import settings

logger = logging.getLogger(__name__)


# ── Stop Words ────────────────────────────────────────────────────────────────
# Stop words are common words that carry no signal for matching.
# We use scikit-learn's built-in English stop word list, but we EXCLUDE
# tech terms that happen to look like stop words (e.g. "c", "r", "go").
#
# WHY NOT REMOVE "c", "r", "go"?
#   These are legitimate programming language names. scikit-learn's stop
#   word list would silently remove them, destroying signal. So we use
#   scikit-learn's list but override it with our own custom set.

# Terms that should NEVER be treated as stop words (they're skill names)
_SKILL_STOP_WORD_EXCEPTIONS = {
    "c", "r", "go", "rust", "sql", "api", "aws", "git",
}

# A minimal, hand-curated stop word list that skips all skill-like tokens
_CUSTOM_STOP_WORDS = [
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
    "for", "of", "with", "by", "from", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "shall",
    "this", "that", "these", "those", "it", "its", "we", "our",
    "you", "your", "they", "their", "i", "my", "me",
    "as", "if", "not", "no", "nor", "so", "yet", "both", "either",
    "about", "above", "after", "before", "between", "into", "through",
    "during", "including", "also", "such", "than", "then", "when",
    "where", "which", "while", "who", "whom", "how", "what",
    "strong", "good", "great", "excellent", "ability", "must", "able",
    "looking", "seeking", "required", "preferred", "plus", "bonus",
    "years", "year", "work", "working", "worked", "join", "team",
    "using", "use", "used", "help", "build", "building", "built",
]


# ── Preprocessor ─────────────────────────────────────────────────────────────

def preprocess_text(text: str) -> str:
    """
    Normalise text before TF-IDF vectorization.

    WHY PREPROCESS?
      Raw resume text has noise that hurts vectorization:
        - Punctuation splits "FastAPI." into "FastAPI" and "" (empty token)
        - Mixed case means "Python" and "python" become separate dimensions
        - Numbers like "5+" or "3rd" add noise without signal

    WHAT WE DO:
      1. Lowercase everything
      2. Remove punctuation (but keep hyphens in compound words)
      3. Collapse whitespace

    WHAT WE DON'T DO:
      We do NOT stem words (turning "running" → "run") because stemming
      destroys tech terms. "Kubernetes" stemmed becomes "kubernet" — not
      in any vocabulary and useless for matching.

    Args:
        text: Raw text string.

    Returns:
        Cleaned, lowercased text string.
    """
    if not text:
        return ""

    # Lowercase
    text = text.lower()

    # Replace hyphens with spaces: "full-stack" → "full stack"
    # Keeps both words as separate tokens instead of one unknown token
    text = text.replace("-", " ")

    # Remove punctuation except apostrophes (preserves "don't" etc.)
    text = text.translate(
        str.maketrans("", "", string.punctuation.replace("'", ""))
    )

    # Collapse multiple whitespace characters into a single space
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ── TF-IDF Scorer ─────────────────────────────────────────────────────────────

def compute_similarity_score(resume_text: str, jd_text: str) -> float:
    """
    Compute cosine similarity between a resume and a job description
    using TF-IDF vectorization.

    HOW IT WORKS STEP BY STEP:
      1. Preprocess both texts (lowercase, remove punctuation)
      2. Feed both into TfidfVectorizer together so they share a vocabulary
      3. Vectorizer produces a 2-row matrix: [resume_vector, jd_vector]
      4. Compute cosine_similarity between the two rows
      5. Result is a float 0.0–1.0

    WHY FIT ON BOTH DOCUMENTS TOGETHER?
      TF-IDF needs to know the "universe" of documents to compute IDF
      (Inverse Document Frequency). With only 2 documents, we fit on both
      so the vocabulary and IDF weights are computed from the same corpus.
      The "rarity" of a term is relative to these 2 documents.

    Args:
        resume_text: Extracted text from the resume PDF.
        jd_text:     Raw job description text.

    Returns:
        Float between 0.0 and 1.0.
        0.0 = no semantic overlap whatsoever
        1.0 = identical content

    Raises:
        ValueError: If either text is empty after preprocessing.
    """
    # ── Step 1: Preprocess ────────────────────────────────────────────────────
    clean_resume = preprocess_text(resume_text)
    clean_jd     = preprocess_text(jd_text)

    if not clean_resume:
        raise ValueError("Resume text is empty after preprocessing.")
    if not clean_jd:
        raise ValueError("Job description text is empty after preprocessing.")

    # ── Step 2: Vectorize ─────────────────────────────────────────────────────
    # TfidfVectorizer parameters explained:
    #
    #   stop_words=_CUSTOM_STOP_WORDS
    #     Remove common words that add noise ("the", "and", "is" etc.)
    #     We use our custom list to protect skill names like "c", "r", "go"
    #
    #   ngram_range=(1, 2)
    #     Capture both single words ("python") AND two-word phrases
    #     ("machine learning", "rest api", "unit testing").
    #     This significantly improves matching for multi-word skills.
    #
    #   min_df=1
    #     Include a term even if it appears in only 1 document.
    #     With only 2 documents, min_df=2 would discard almost everything.
    #
    #   max_features=10000
    #     Cap vocabulary size to prevent memory issues on very long texts.
    #     10k features is more than enough for any resume/JD pair.
    #
    #   sublinear_tf=True
    #     Apply log normalization to term frequency: TF = 1 + log(TF)
    #     Prevents a word repeated 100 times from dominating the vector.
    #     Standard practice in information retrieval systems.

    vectorizer = TfidfVectorizer(
        stop_words=_CUSTOM_STOP_WORDS,
        ngram_range=(1, 2),
        min_df=1,
        max_features=10_000,
        sublinear_tf=True,
    )

    # ── Step 3: Fit and transform ─────────────────────────────────────────────
    # fit_transform() on a list of 2 strings returns a (2 x vocab_size) matrix
    # Row 0 = resume vector, Row 1 = JD vector
    try:
        tfidf_matrix = vectorizer.fit_transform([clean_resume, clean_jd])
    except ValueError as e:
        # This happens if the vocabulary ends up empty (e.g. all stop words)
        logger.error(f"TF-IDF vectorization failed: {e}")
        raise ValueError(
            f"Could not vectorize the provided texts. "
            f"Ensure both resume and job description contain meaningful content. "
            f"Details: {str(e)}"
        )

    # ── Step 4: Compute cosine similarity ─────────────────────────────────────
    # cosine_similarity returns a 2x2 matrix:
    #   [[sim(resume, resume),  sim(resume, jd)  ],
    #    [sim(jd, resume),      sim(jd, jd)      ]]
    #
    # We want [0][1]: similarity of resume (row 0) with JD (row 1)
    similarity_matrix = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])
    raw_score = float(similarity_matrix[0][0])

    # ── Step 5: Clamp and round ───────────────────────────────────────────────
    # Cosine similarity is theoretically 0.0–1.0 but floating point arithmetic
    # can produce values like 1.0000000002 — clamp to be safe.
    score = round(max(0.0, min(1.0, raw_score)), 4)

    logger.info(f"TF-IDF cosine similarity score: {score}")
    return score


# ── Verdict Generator ─────────────────────────────────────────────────────────

def generate_verdict(score: float) -> str:
    """
    Convert a numeric similarity score into a human-readable hiring verdict.

    Thresholds are defined in config.py so they can be tuned in one place.
    The values were chosen based on typical cosine similarity distributions
    for resume/JD pairs — most real matches cluster between 0.3 and 0.8.

    Args:
        score: Float between 0.0 and 1.0 from compute_similarity_score().

    Returns:
        One of: "Strong Fit", "Partial Fit", "Weak Fit"
    """
    if score >= settings.STRONG_FIT_THRESHOLD:
        return "Strong Fit"
    elif score >= settings.PARTIAL_FIT_THRESHOLD:
        return "Partial Fit"
    else:
        return "Weak Fit"


# ── Score Breakdown ───────────────────────────────────────────────────────────

def get_score_breakdown(score: float) -> dict:
    """
    Return a structured breakdown of what the score means.

    Useful for the API response — gives context beyond a raw number.
    Think of this as the "explain your answer" layer.

    Args:
        score: Float between 0.0 and 1.0.

    Returns:
        Dict with verdict, percentage, confidence label, and advice.
    """
    verdict    = generate_verdict(score)
    percentage = round(score * 100, 1)

    # Human-readable advice per verdict tier
    advice_map = {
        "Strong Fit": (
            "This candidate strongly matches the job requirements. "
            "Recommended for interview."
        ),
        "Partial Fit": (
            "This candidate partially matches the job requirements. "
            "Consider if missing skills can be learned on the job."
        ),
        "Weak Fit": (
            "This candidate has limited overlap with the job requirements. "
            "Significant skill gaps detected."
        ),
    }

    return {
        "score":      score,
        "percentage": percentage,
        "verdict":    verdict,
        "advice":     advice_map[verdict],
    }
