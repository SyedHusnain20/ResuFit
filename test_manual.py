import sys
sys.path.insert(0, ".")

from app.services.skill_extractor import extract_skills, compare_skills
from app.services.scorer import compute_similarity_score, get_score_breakdown
from app.services.analyzer import compute_hybrid_score

resume_text = """
Senior Python Developer with 4 years of professional experience building
scalable backend systems. Proficient in FastAPI and Flask for REST API development.
Extensive experience deploying containerized applications using Docker and Kubernetes
on AWS infrastructure. Strong database skills with PostgreSQL and Redis.
Applied machine learning models using scikit-learn and PyTorch in production.
Practiced test-driven development using pytest and GitHub Actions for CI/CD.
"""

jd_text = """
We are hiring a Python Backend Engineer.
You will build REST APIs using FastAPI or Django.
Strong experience with Docker and AWS is required.
You must be comfortable with PostgreSQL databases.
Experience with machine learning and scikit-learn is a plus.
We use Git and GitHub Actions. pytest knowledge required.
"""

resume_skills    = extract_skills(resume_text)
jd_skills        = extract_skills(jd_text)
comparison       = compare_skills(resume_skills, jd_skills)
tfidf_score      = compute_similarity_score(resume_text, jd_text)
skill_match_rate = comparison["match_rate"]
hybrid_score     = compute_hybrid_score(tfidf_score, skill_match_rate)
breakdown        = get_score_breakdown(hybrid_score)

print("=" * 50)
print("TF-IDF Score:     ", tfidf_score, " (40% weight)")
print("Skill Match Rate: ", round(skill_match_rate, 4), " (60% weight)")
print("Hybrid Score:     ", hybrid_score)
print("Fit Percentage:   ", breakdown["percentage"], "%")
print("Verdict:          ", breakdown["verdict"])
print("Matched Skills:   ", comparison["matched_skills"])
print("Missing Skills:   ", comparison["missing_skills"])
print("=" * 50)