"""
test_taxonomy.py — Skills Taxonomy Integrity Tests
────────────────────────────────────────────────────
WHY TEST THE TAXONOMY?
  The taxonomy is the foundation of skill extraction — if it has data
  integrity problems, every analysis is wrong. These tests are like
  database constraint checks: they verify the data is self-consistent.

WHAT WE CHECK:
  1. No duplicate skills across categories (a skill in two categories
     would cause non-deterministic category assignment)
  2. No empty categories (they'd show up as empty lists in responses)
  3. All skills are lowercase (the extractor lowercases input text —
     if taxonomy skills have uppercase, they'll never match)
  4. No leading/trailing whitespace in skill strings
  5. ALL_SKILLS and SKILL_TO_CATEGORY are consistent with SKILLS_TAXONOMY
  6. Reverse lookup returns correct categories
  7. Critical skills we depend on are actually in the taxonomy
"""

import pytest

from app.services.skills_taxonomy import (
    ALL_SKILLS,
    SKILL_TO_CATEGORY,
    SKILLS_TAXONOMY,
)


class TestTaxonomyStructure:

    def test_taxonomy_is_not_empty(self):
        assert len(SKILLS_TAXONOMY) > 0

    def test_all_categories_have_skills(self):
        for category, skills in SKILLS_TAXONOMY.items():
            assert len(skills) > 0, f"Category '{category}' is empty"

    def test_minimum_category_count(self):
        # We defined 12 categories — guard against accidental deletion
        assert len(SKILLS_TAXONOMY) >= 10

    def test_minimum_total_skill_count(self):
        # We should have at least 100 skills across all categories
        assert len(ALL_SKILLS) >= 100

    def test_no_duplicate_skills_across_categories(self):
        """
        If 'python' appears in both 'Programming Languages' AND 'Web Frameworks',
        SKILL_TO_CATEGORY can only store one mapping — the other is silently lost.
        This test catches that before it causes subtle bugs.
        """
        seen = {}
        duplicates = []

        for category, skills in SKILLS_TAXONOMY.items():
            for skill in skills:
                if skill in seen:
                    duplicates.append(
                        f"'{skill}' in both '{seen[skill]}' and '{category}'"
                    )
                else:
                    seen[skill] = category

        assert duplicates == [], (
            f"Duplicate skills found across categories:\n"
            + "\n".join(duplicates)
        )

    def test_all_skills_are_lowercase(self):
        """
        The extractor lowercases input text before matching.
        If a skill is 'FastAPI' (uppercase), it will never match 'fastapi'.
        All taxonomy skills must be lowercase.
        """
        non_lowercase = [
            skill
            for skills in SKILLS_TAXONOMY.values()
            for skill in skills
            if skill != skill.lower()
        ]
        assert non_lowercase == [], (
            f"Skills with uppercase letters found: {non_lowercase}"
        )

    def test_no_skills_with_leading_trailing_whitespace(self):
        """
        A skill like ' python' (with a leading space) would never match
        because the regex pattern would include the space as part of the
        word boundary check.
        """
        whitespace_skills = [
            skill
            for skills in SKILLS_TAXONOMY.values()
            for skill in skills
            if skill != skill.strip()
        ]
        assert whitespace_skills == [], (
            f"Skills with whitespace found: {whitespace_skills}"
        )

    def test_no_empty_skill_strings(self):
        empty_skills = [
            skill
            for skills in SKILLS_TAXONOMY.values()
            for skill in skills
            if not skill
        ]
        assert empty_skills == []


class TestDerivedDataStructures:
    """
    ALL_SKILLS and SKILL_TO_CATEGORY are derived from SKILLS_TAXONOMY.
    These tests verify the derivation is correct and consistent.
    """

    def test_all_skills_contains_every_taxonomy_skill(self):
        for category, skills in SKILLS_TAXONOMY.items():
            for skill in skills:
                assert skill in ALL_SKILLS, (
                    f"'{skill}' from '{category}' missing from ALL_SKILLS"
                )

    def test_all_skills_count_matches_taxonomy(self):
        # Count unique skills in taxonomy
        taxonomy_skill_count = len({
            skill
            for skills in SKILLS_TAXONOMY.values()
            for skill in skills
        })
        assert len(ALL_SKILLS) == taxonomy_skill_count

    def test_skill_to_category_covers_all_skills(self):
        for skill in ALL_SKILLS:
            assert skill in SKILL_TO_CATEGORY, (
                f"'{skill}' in ALL_SKILLS but missing from SKILL_TO_CATEGORY"
            )

    def test_skill_to_category_maps_to_valid_categories(self):
        valid_categories = set(SKILLS_TAXONOMY.keys())
        for skill, category in SKILL_TO_CATEGORY.items():
            assert category in valid_categories, (
                f"'{skill}' maps to unknown category '{category}'"
            )

    def test_reverse_lookup_is_correct(self):
        # For every skill in every category, the reverse lookup must agree
        for category, skills in SKILLS_TAXONOMY.items():
            for skill in skills:
                assert SKILL_TO_CATEGORY[skill] == category, (
                    f"'{skill}' maps to '{SKILL_TO_CATEGORY[skill]}' "
                    f"but should map to '{category}'"
                )


class TestCriticalSkillsPresent:
    """
    Smoke tests: verify that the most important skills for a backend AI/ML
    portfolio project are actually in the taxonomy. If someone accidentally
    deletes a category, these tests catch it.
    """

    @pytest.mark.parametrize("skill", [
        "python", "javascript", "typescript", "go", "java",
    ])
    def test_core_languages_present(self, skill):
        assert skill in ALL_SKILLS, f"Critical language '{skill}' missing"

    @pytest.mark.parametrize("skill", [
        "fastapi", "django", "flask", "react", "spring boot",
    ])
    def test_core_frameworks_present(self, skill):
        assert skill in ALL_SKILLS, f"Critical framework '{skill}' missing"

    @pytest.mark.parametrize("skill", [
        "docker", "kubernetes", "aws", "git", "github actions",
    ])
    def test_core_devops_present(self, skill):
        assert skill in ALL_SKILLS, f"Critical DevOps skill '{skill}' missing"

    @pytest.mark.parametrize("skill", [
        "postgresql", "mongodb", "redis", "mysql",
    ])
    def test_core_databases_present(self, skill):
        assert skill in ALL_SKILLS, f"Critical database '{skill}' missing"

    @pytest.mark.parametrize("skill", [
        "machine learning", "scikit-learn", "pytorch", "tensorflow",
        "natural language processing", "deep learning",
    ])
    def test_core_ml_skills_present(self, skill):
        assert skill in ALL_SKILLS, f"Critical ML skill '{skill}' missing"

    @pytest.mark.parametrize("skill", [
        "pytest", "git", "rest", "sql", "agile",
    ])
    def test_common_resume_skills_present(self, skill):
        assert skill in ALL_SKILLS, f"Common resume skill '{skill}' missing"
