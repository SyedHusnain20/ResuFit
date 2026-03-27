"""
config.py — Centralized Application Settings
─────────────────────────────────────────────
All tuneable values live here. Change a threshold, limit, or label
in one place instead of hunting through the codebase.
"""


class Settings:
    # ── App Metadata ──────────────────────────────────────────────────────────
    APP_NAME: str = "ResuFit"
    APP_VERSION: str = "1.0.0"
    APP_DESCRIPTION: str = """
## AI-Powered Resume Screening API

ResuFit analyzes a resume PDF against a job description and returns a structured
fit report using a **hybrid AI scoring model**.

---

### How Scoring Works

ResuFit combines two independent signals:

| Signal | Weight | What It Captures |
|---|---|---|
| **TF-IDF Cosine Similarity** | 40% | Semantic alignment, seniority language, domain vocabulary |
| **Skill Match Rate** | 60% | Exact technical skill matches from a 500+ skill taxonomy |

```
hybrid_score = (tfidf_score × 0.40) + (skill_match_rate × 0.60)
```

### Verdict Thresholds

| Score | Verdict |
|---|---|
| ≥ 0.65 | ✅ Strong Fit — Recommended for interview |
| 0.40 – 0.64 | 🟡 Partial Fit — Consider skill gaps |
| < 0.40 | ❌ Weak Fit — Significant gaps detected |

---

### Requirements
- Resume must be a **text-based PDF** (not a scanned image)
- Maximum file size: **5 MB**
- Job description minimum: **50 characters**

### Error Response Shape
All errors follow a consistent format:
```json
{ "detail": "Human-readable description of what went wrong." }
```

### Request Tracing
Every response includes an `X-Request-ID` header — an 8-character unique ID
you can use to correlate your request with server logs.
    """

    # ── File Upload Constraints ───────────────────────────────────────────────
    MAX_FILE_SIZE_BYTES: int = 5 * 1024 * 1024  # 5 MB
    ALLOWED_CONTENT_TYPES: list[str] = ["application/pdf"]

    # ── Scoring Thresholds ────────────────────────────────────────────────────
    STRONG_FIT_THRESHOLD: float = 0.65
    PARTIAL_FIT_THRESHOLD: float = 0.40


# Module-level singleton — import this everywhere
settings = Settings()
