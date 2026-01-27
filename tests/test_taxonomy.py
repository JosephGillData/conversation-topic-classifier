"""
Taxonomy Validation Tests

This module contains tests to validate the taxonomy CSV file before running
the classifier. These tests ensure the taxonomy meets the requirements for
the structured output classifier to work correctly.

Run tests with:
    pytest tests/test_taxonomy.py -v
    make test-taxonomy

Why these tests matter:
    - The classifier uses Pydantic Literal types from taxonomy names
    - Duplicate or blank names would cause runtime errors
    - Missing catch-all category leads to poor handling of edge cases
    - Empty descriptions reduce classification accuracy

Test categories:
    1. File structure tests (columns, non-empty)
    2. Data quality tests (uniqueness, no blanks)
    3. Best practice tests (catch-all exists, reasonable count)
"""

import pandas as pd
import pytest


# Path to the taxonomy file (relative to project root)
TAXONOMY_PATH = "data/taxonomy.csv"


# =============================================================================
# File Structure Tests
# =============================================================================

def test_taxonomy_file_exists():
    """
    Verify the taxonomy CSV file exists and is readable.

    This is a prerequisite for all other tests. If the file doesn't exist,
    the classifier will fail immediately on startup.
    """
    df = pd.read_csv(TAXONOMY_PATH)
    assert df is not None, "Failed to read taxonomy.csv"


def test_taxonomy_has_required_columns():
    """
    Verify taxonomy has the required columns: taxonomy_name and taxonomy_description.

    - taxonomy_name: Used as the Literal type values and output labels
    - taxonomy_description: Injected into the prompt to guide classification

    The classifier will raise ValueError if these columns are missing.
    """
    df = pd.read_csv(TAXONOMY_PATH)
    required = {"taxonomy_name", "taxonomy_description"}
    missing = required - set(df.columns)
    assert len(missing) == 0, f"Missing required columns: {missing}"


def test_taxonomy_not_empty():
    """
    Verify taxonomy has at least one category defined.

    An empty taxonomy would result in an invalid Pydantic Literal type
    and the classifier would have no valid output options.
    """
    df = pd.read_csv(TAXONOMY_PATH)
    assert len(df) > 0, "Taxonomy is empty - at least one category required"


# =============================================================================
# Data Quality Tests
# =============================================================================

def test_taxonomy_names_unique():
    """
    Verify all taxonomy names are unique.

    Duplicate names would cause issues because:
    - Pydantic Literal types require unique values
    - Output analysis would conflate different categories
    - The LLM might inconsistently choose between duplicates
    """
    df = pd.read_csv(TAXONOMY_PATH)
    names = df["taxonomy_name"].tolist()
    duplicates = [name for name in set(names) if names.count(name) > 1]
    assert len(duplicates) == 0, f"Duplicate taxonomy names found: {duplicates}"


def test_taxonomy_names_not_blank():
    """
    Verify no taxonomy names are empty or whitespace-only.

    Blank names would:
    - Create invalid Literal type values
    - Be confusing in output analysis
    - Potentially cause matching issues in the LLM
    """
    df = pd.read_csv(TAXONOMY_PATH)
    names = df["taxonomy_name"].astype(str).str.strip().tolist()
    blank_indices = [i + 1 for i, name in enumerate(names) if not name]  # +1 for 1-indexed rows
    assert len(blank_indices) == 0, f"Blank taxonomy names at CSV rows: {blank_indices}"


def test_taxonomy_descriptions_not_empty():
    """
    Verify each taxonomy has a non-empty description.

    Descriptions are critical because they:
    - Guide the LLM on what each category means
    - Define category boundaries (Includes/Excludes)
    - Improve classification accuracy significantly

    Categories without descriptions will still work but may be misclassified.
    """
    df = pd.read_csv(TAXONOMY_PATH)
    df["taxonomy_description"] = df["taxonomy_description"].fillna("").astype(str).str.strip()
    empty_desc_categories = df[df["taxonomy_description"] == ""]["taxonomy_name"].tolist()
    assert len(empty_desc_categories) == 0, (
        f"Categories missing descriptions: {empty_desc_categories}. "
        "Add descriptions to improve classification accuracy."
    )


# =============================================================================
# Best Practice Tests
# =============================================================================

def test_general_category_exists():
    """
    Verify a catch-all category exists for ambiguous conversations.

    The 'General Enquiries & Multi-Intent' category (or similar) is important because:
    - Some conversations don't fit any specific category
    - Multi-intent conversations need a fallback
    - It prevents forcing bad classifications

    Without this, the model may force-fit ambiguous cases into wrong categories.
    """
    df = pd.read_csv(TAXONOMY_PATH)
    names = df["taxonomy_name"].tolist()

    # Check for the standard catch-all category name
    has_catchall = "General Enquiries & Multi-Intent" in names

    assert has_catchall, (
        "Missing catch-all category 'General Enquiries & Multi-Intent'. "
        "This category handles ambiguous or multi-intent conversations. "
        "Consider adding it to improve classification quality."
    )


def test_reasonable_category_count():
    """
    Verify taxonomy has a reasonable number of categories (2-50).

    Guidelines:
    - < 2 categories: Not enough granularity for useful classification
    - 2-15 categories: Typical for focused use cases
    - 15-30 categories: Common for comprehensive taxonomies
    - 30-50 categories: Complex but manageable
    - > 50 categories: May indicate over-specification; consider hierarchy

    Very large taxonomies can:
    - Exceed context limits when injected into prompts
    - Reduce classification accuracy (too many similar options)
    - Make analysis and reporting unwieldy
    """
    df = pd.read_csv(TAXONOMY_PATH)
    count = len(df)

    assert count >= 2, (
        f"Only {count} category found. "
        "At least 2 categories needed for meaningful classification."
    )

    assert count <= 50, (
        f"Found {count} categories (>50). "
        "Consider consolidating into a hierarchical taxonomy or reducing granularity."
    )


# =============================================================================
# Optional: Advanced Validation (uncomment if needed)
# =============================================================================

# def test_taxonomy_names_valid_characters():
#     """Verify taxonomy names don't contain problematic characters."""
#     df = pd.read_csv(TAXONOMY_PATH)
#     import re
#     for name in df["taxonomy_name"]:
#         # Allow letters, numbers, spaces, ampersands, commas, hyphens
#         assert re.match(r'^[\w\s&,\-]+$', str(name)), (
#             f"Invalid characters in taxonomy name: '{name}'"
#         )

# def test_descriptions_have_minimum_length():
#     """Verify descriptions are substantive (at least 50 characters)."""
#     df = pd.read_csv(TAXONOMY_PATH)
#     short_descs = df[df["taxonomy_description"].str.len() < 50]["taxonomy_name"].tolist()
#     assert len(short_descs) == 0, (
#         f"Categories with very short descriptions (<50 chars): {short_descs}"
#     )
