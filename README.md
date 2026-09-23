# ResuFit 🎯
### AI-Powered Resume Screening API

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green.svg)](https://fastapi.tiangolo.com)
[![Tests](https://img.shields.io/badge/Tests-205%20passing-brightgreen.svg)](#testing)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

ResuFit is a production-ready REST API that screens resumes against job descriptions using a **hybrid AI scoring model** combining TF-IDF semantic similarity and keyword-based skill extraction.

Upload a resume PDF and a job description → receive a structured fit report in milliseconds.

**Live API:** `https://resufit.onrender.com` *(update after deployment)*
**Interactive Docs:** `https://resufit.onrender.com/docs`

---

## Demo

```bash
curl -X POST "https://resufit.onrender.com/api/v1/analyze" \
  -F "resume=@alex_johnson_resume.pdf" \
  -F "job_description=We are hiring a Python Backend Engineer with FastAPI, Docker, and AWS experience..."
```

**Response:**
```json
{
  "fit_score": 0.6513,
  "fit_percentage": 65.1,
  "verdict": "Strong Fit",
  "advice": "This candidate strongly matches the job requirements. Recommended for interview.",
  "matched_skills": ["python", "fastapi", "docker", "aws", "postgresql", "machine learning"],
  "missing_skills": ["scikit-learn"],
  "extra_skills": ["flask", "redis", "pytorch", "kubernetes"],
  "tfidf_score": 0.1964,
  "skill_score": 0.9545,
  "resume_skill_count": 44,
  "jd_skill_count": 22,
  "resume_text_length": 3111
}
```

---

## How It Works

ResuFit uses a **two-signal hybrid scoring model** designed specifically for technical job matching:

### Signal 1 — TF-IDF Semantic Similarity (40% weight)
Converts both texts into TF-IDF vectors and computes cosine similarity. Captures:
- Overall semantic alignment between resume and JD
- Seniority language ("architected", "led a team of", "junior developer")
- Domain vocabulary ("distributed systems", "real-time processing")
- Contextual soft skills ("cross-functional", "stakeholder management")

### Signal 2 — Skill Keyword Extraction (60% weight)
Matches skills against a curated taxonomy of **500+ technical skills** across 12 categories using boundary-aware regex patterns. Returns:
- Exact matched skills (present in both resume and JD)
- Missing skills (in JD but absent from resume — the candidate's gaps)
- Extra skills (additional value the candidate brings)

### Hybrid Score Formula
```
hybrid_score = (tfidf_score × 0.40) + (skill_match_rate × 0.60)
```

Skills get 60% weight because for technical roles, specific technology matches ("Docker", "FastAPI", "PostgreSQL") are more predictive of job fit than semantic writing similarity.

### Verdict Thresholds

| Score Range | Verdict |
|---|---|
| ≥ 0.65 | ✅ Strong Fit |
| 0.40 – 0.64 | 🟡 Partial Fit |
| < 0.40 | ❌ Weak Fit |

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Web Framework | FastAPI 0.111 | Async REST API, auto-generated docs |
| ML Scoring | Scikit-learn (TF-IDF + Cosine Similarity) | Semantic resume-JD matching |
| PDF Parsing | pdfplumber | Text extraction from resume PDFs |
| Validation | Pydantic v2 | Request/response schema validation |
| Skill Extraction | Custom regex taxonomy | 500+ skills across 12 categories |
| Testing | Pytest (205 tests) | Unit, integration, and scenario tests |
| Deployment | Render | Cloud hosting with auto-deploy |

---

## API Reference

### `POST /api/v1/analyze`

Analyze a resume PDF against a job description.

**Request** — `multipart/form-data`

| Field | Type | Required | Description |
|---|---|---|---|
| `resume` | File (PDF) | ✅ | Text-based PDF, max 5 MB |
| `job_description` | string | ✅ | Min 50 characters |

**Response — 200 OK**

| Field | Type | Description |
|---|---|---|
| `fit_score` | float | Hybrid score 0.0–1.0 |
| `fit_percentage` | float | fit_score × 100 |
| `tfidf_score` | float | Raw TF-IDF cosine similarity |
| `skill_score` | float | Skill match rate 0.0–1.0 |
| `verdict` | string | Strong Fit / Partial Fit / Weak Fit |
| `advice` | string | Hiring recommendation |
| `matched_skills` | list[str] | Skills in both resume and JD |
| `missing_skills` | list[str] | Skills in JD but not in resume |
| `extra_skills` | list[str] | Skills in resume not in JD |
| `resume_skill_count` | int | Total skills found in resume |
| `jd_skill_count` | int | Total skills found in JD |
| `resume_text_length` | int | Character count of extracted text |

**Error Responses**

| Status | Meaning |
|---|---|
| `400` | Invalid PDF, corrupt file, or no extractable text |
| `413` | File exceeds 5 MB limit |
| `422` | Missing field or job description too short |
| `500` | Unexpected server error |

All errors: `{ "detail": "Human-readable message" }`

---

## Skill Taxonomy

ResuFit detects **500+ skills** across 12 categories:

| Category | Examples |
|---|---|
| Programming Languages | Python, JavaScript, TypeScript, Go, Rust, Java |
| Web Frameworks | FastAPI, Django, Flask, React, Next.js, Spring Boot |
| Databases | PostgreSQL, MongoDB, Redis, MySQL, Elasticsearch |
| Cloud & DevOps | AWS, Docker, Kubernetes, Terraform, GitHub Actions |
| Machine Learning & AI | scikit-learn, PyTorch, TensorFlow, HuggingFace, BERT |
| Data Science | pandas, numpy, Spark, Airflow, Tableau |
| APIs & Integration | REST, GraphQL, gRPC, Kafka, RabbitMQ |
| Testing | pytest, Jest, Cypress, TDD, Selenium |
| Version Control | Git, GitHub, GitLab, Agile, Scrum |
| Security | OAuth, JWT, TLS, OWASP |
| Mobile | React Native, Flutter, iOS, Android |
| Soft Skills | Leadership, Communication, Mentoring |

---

## Project Structure

```
resufit/
├── app/
│   ├── main.py                # FastAPI app — middleware + route registration
│   ├── config.py              # Centralized settings (thresholds, limits)
│   ├── exceptions.py          # Global exception handlers + logging config
│   ├── middleware/
│   │   └── logging.py         # Request logging with unique request IDs
│   ├── routers/
│   │   └── analyze.py         # POST /api/v1/analyze endpoint
│   ├── schemas/
│   │   └── analysis.py        # Pydantic request/response models
│   └── services/
│       ├── analyzer.py        # Main pipeline orchestrator
│       ├── pdf_parser.py      # PDF text extraction (pdfplumber)
│       ├── skill_extractor.py # Regex-based skill matching engine
│       ├── skills_taxonomy.py # 500+ skills across 12 categories
│       └── scorer.py          # TF-IDF vectorizer + cosine similarity
├── tests/                     # 205 tests across 8 test files
├── conftest.py                # Shared pytest fixtures
├── pytest.ini                 # Pytest configuration
└── requirements.txt
```

---

## Local Development

### Prerequisites
- Python 3.10+

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/resufit.git
cd resufit

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start the development server
uvicorn app.main:app --reload
```

Visit **`http://localhost:8000/docs`** for the interactive Swagger UI.

### Running Tests

```bash
# Run all 205 tests
python -m pytest

# Run a specific module
python -m pytest tests/test_scorer.py -v

# Run only scenario/integration tests
python -m pytest -m integration

# Run with short tracebacks
python -m pytest --tb=short
```

---

## Design Decisions

**Why keyword matching instead of spaCy NER?**
spaCy's NER doesn't recognize "FastAPI" or "TF-IDF" — they're not in any general-purpose NLP vocabulary. A curated keyword taxonomy gives deterministic, explainable results. In hiring tools, every decision must be justifiable.

**Why 60% weight on skills vs 40% on TF-IDF?**
For technical roles, specific tool matches are more predictive of fit than writing similarity. A resume saying "orchestrated containerized deployments" scores high on TF-IDF but might not mention Docker. Skills get higher weight for precision.

**Why lookahead/lookbehind instead of `\b` word boundaries?**
`\b` fails for skills ending in non-word characters like `c++`. `(?<!\w)` and `(?!\w)` work universally regardless of whether the skill starts or ends with a symbol.

**Why a `dataclass` for `AnalysisResult`?**
Typed fields, IDE autocomplete, and a clear contract. The Pydantic schema in Step 6 mirrors it exactly, adding validation before the response leaves the server.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Contact

**Engr. Hasnain Zainulabdin**
R&R Digital Solutions

Contact: 03126641281 | [HasnainZainulabdin@gmail.com](mailto:HasnainZainulabdin@gmail.com)
Website: https://hasnainzainulabdin.vercel.app/

---

*Built as a portfolio project demonstrating AI/ML API development with Python and FastAPI.*
