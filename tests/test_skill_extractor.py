"""
test_skill_extractor.py — Unit Tests for Skill Extractor
──────────────────────────────────────────────────────────
These tests focus on the trickiest parts of keyword extraction:
  - Word boundary matching (does "go" match inside "django"?)
  - Case insensitivity (PYTHON vs python vs Python)
  - Multi-word skills ("machine learning", "spring boot")
  - Empty and edge-case inputs
  - Comparison logic (matched, missing, extra)
"""

import pytest

from app.services.skill_extractor import (
    compare_skills,
    extract_skills,
    get_skills_by_category,
)


# ── extract_skills Tests ──────────────────────────────────────────────────────

class TestExtractSkills:

    def test_extracts_simple_skill(self):
        text = "I have 3 years of experience with Python."
        result = extract_skills(text)
        assert "python" in result

    def test_case_insensitive_matching(self):
        # All three variants must be detected
        assert "python" in extract_skills("I know PYTHON")
        assert "python" in extract_skills("I know Python")
        assert "python" in extract_skills("I know python")

    def test_multi_word_skill(self):
        text = "Experience in machine learning and deep learning projects."
        result = extract_skills(text)
        assert "machine learning" in result
        assert "deep learning" in result

    def test_word_boundary_prevents_false_match(self):
        # "go" must NOT match inside "django" or "good" or "google"
        text = "I use Django for good projects at Google."
        result = extract_skills(text)
        # "django" should match as its own skill
        assert "django" in result  
        # but standalone "go" must NOT match because there's no isolated "go"
        assert "go" not in result

    def test_r_language_not_matched_in_other_words(self):
        # "r" is a skill but must not match inside "developer" or "your"
        text = "I am a developer who loves your work."
        result = extract_skills(text)
        assert "r" not in result

    def test_r_language_matched_when_standalone(self):
        # "R" as a standalone word/token should match
        text = "Proficient in R and Python for statistical analysis."
        result = extract_skills(text)
        assert "r" in result

    def test_cpp_skill_matched(self):
        # "c++" contains special regex characters — our escaping must handle it
        text = "Worked extensively with C++ and Java."
        result = extract_skills(text)
        assert "c++" in result

    def test_multiple_skills_extracted(self):
        text = (
            "Built REST APIs using FastAPI and deployed on AWS with Docker. "
            "Used PostgreSQL as the database and Redis for caching."
        )
        result = extract_skills(text)
        assert "fastapi" in result
        assert "aws" in result
        assert "docker" in result
        assert "postgresql" in result
        assert "redis" in result

    def test_empty_string_returns_empty_set(self):
        assert extract_skills("") == set()

    def test_whitespace_only_returns_empty_set(self):
        assert extract_skills("   \n\t  ") == set()

    def test_no_skills_text_returns_empty_set(self):
        text = "The weather is nice today. I enjoy long walks."
        result = extract_skills(text)
        assert isinstance(result, set)
        assert len(result) == 0

    def test_returns_set_type(self):
        result = extract_skills("Python developer")
        assert isinstance(result, set)

    def test_no_duplicate_skills(self):
        # Even if "python" appears many times, it should appear once in the set
        text = "Python Python Python developer with Python skills in Python."
        result = extract_skills(text)
        # Sets don't allow duplicates — this is implicit, but good to assert
        assert len([s for s in result if s == "python"]) == 1


# ── compare_skills Tests ──────────────────────────────────────────────────────

class TestCompareSkills:

    def test_perfect_match(self):
        resume = {"python", "fastapi", "docker"}
        jd     = {"python", "fastapi", "docker"}
        result = compare_skills(resume, jd)
        assert set(result["matched_skills"]) == {"python", "fastapi", "docker"}
        assert result["missing_skills"] == []
        assert result["match_rate"] == 1.0

    def test_partial_match(self):
        resume = {"python", "fastapi"}
        jd     = {"python", "fastapi", "docker", "kubernetes"}
        result = compare_skills(resume, jd)
        assert set(result["matched_skills"]) == {"python", "fastapi"}
        assert set(result["missing_skills"]) == {"docker", "kubernetes"}
        assert result["match_rate"] == 0.5

    def test_no_match(self):
        resume = {"java", "spring boot"}
        jd     = {"python", "fastapi"}
        result = compare_skills(resume, jd)
        assert result["matched_skills"] == []
        assert set(result["missing_skills"]) == {"python", "fastapi"}
        assert result["match_rate"] == 0.0

    def test_extra_skills_detected(self):
        # Resume has skills the JD doesn't ask for
        resume = {"python", "fastapi", "rust", "haskell"}
        jd     = {"python", "fastapi"}
        result = compare_skills(resume, jd)
        assert set(result["extra_skills"]) == {"rust", "haskell"}

    def test_empty_jd_skills(self):
        # Edge case: JD has no recognisable skills — match_rate should be 0
        resume = {"python", "fastapi"}
        jd     = set()
        result = compare_skills(resume, jd)
        assert result["match_rate"] == 0.0
        assert result["missing_skills"] == []

    def test_output_is_sorted(self):
        # Sorted output ensures the API response is deterministic
        resume = {"python", "docker", "aws", "fastapi"}
        jd     = {"python", "docker", "aws", "fastapi"}
        result = compare_skills(resume, jd)
        assert result["matched_skills"] == sorted(result["matched_skills"])

    def test_match_rate_precision(self):
        resume = {"python"}
        jd     = {"python", "fastapi", "docker"}
        result = compare_skills(resume, jd)
        # 1 matched out of 3 = 0.3333
        assert abs(result["match_rate"] - 0.3333) < 0.001


# ── get_skills_by_category Tests ─────────────────────────────────────────────

class TestGetSkillsByCategory:

    def test_groups_skills_correctly(self):
        skills = {"python", "fastapi", "docker"}
        result = get_skills_by_category(skills)
        # Each category must be a list
        for category, skills_list in result.items():
            assert isinstance(skills_list, list)

    def test_python_in_programming_languages(self):
        skills = {"python"}
        result = get_skills_by_category(skills)
        assert "Programming Languages" in result
        assert "python" in result["Programming Languages"]

    def test_docker_in_cloud_devops(self):
        skills = {"docker"}
        result = get_skills_by_category(skills)
        assert "Cloud & DevOps" in result

    def test_empty_set_returns_empty_dict(self):
        result = get_skills_by_category(set())
        assert result == {}

    def test_skills_within_category_are_sorted(self):
        skills = {"python", "javascript", "typescript", "go"}
        result = get_skills_by_category(skills)
        for skill_list in result.values():
            assert skill_list == sorted(skill_list)
