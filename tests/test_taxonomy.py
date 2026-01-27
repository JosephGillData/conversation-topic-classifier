"""Tests for taxonomy validation."""

import pandas as pd
import pytest


TAXONOMY_PATH = "data/taxonomy.csv"


def test_taxonomy_file_exists():
    """Taxonomy CSV file should exist."""
    df = pd.read_csv(TAXONOMY_PATH)
    assert df is not None


def test_taxonomy_has_required_columns():
    """Taxonomy must have taxonomy_name and taxonomy_description columns."""
    df = pd.read_csv(TAXONOMY_PATH)
    required = {"taxonomy_name", "taxonomy_description"}
    assert required.issubset(set(df.columns)), f"Missing columns: {required - set(df.columns)}"


def test_taxonomy_names_unique():
    """All taxonomy names must be unique."""
    df = pd.read_csv(TAXONOMY_PATH)
    names = df["taxonomy_name"].tolist()
    duplicates = [name for name in names if names.count(name) > 1]
    assert len(names) == len(set(names)), f"Duplicate taxonomy names found: {set(duplicates)}"


def test_taxonomy_not_empty():
    """Taxonomy must have at least one category."""
    df = pd.read_csv(TAXONOMY_PATH)
    assert len(df) > 0, "Taxonomy is empty"


def test_taxonomy_names_not_blank():
    """Taxonomy names should not be empty or whitespace."""
    df = pd.read_csv(TAXONOMY_PATH)
    names = df["taxonomy_name"].astype(str).str.strip().tolist()
    blank_names = [i for i, name in enumerate(names) if not name]
    assert len(blank_names) == 0, f"Blank taxonomy names at rows: {blank_names}"


def test_general_category_exists():
    """Catch-all category should exist for ambiguous cases."""
    df = pd.read_csv(TAXONOMY_PATH)
    names = df["taxonomy_name"].tolist()
    assert "General Enquiries & Multi-Intent" in names, (
        "Missing catch-all category 'General Enquiries & Multi-Intent'. "
        "Consider adding one to handle ambiguous conversations."
    )


def test_taxonomy_descriptions_not_empty():
    """Each taxonomy should have a non-empty description."""
    df = pd.read_csv(TAXONOMY_PATH)
    df["taxonomy_description"] = df["taxonomy_description"].fillna("").astype(str).str.strip()
    empty_descs = df[df["taxonomy_description"] == ""]["taxonomy_name"].tolist()
    assert len(empty_descs) == 0, f"Categories missing descriptions: {empty_descs}"


def test_reasonable_category_count():
    """Taxonomy should have a reasonable number of categories (2-50)."""
    df = pd.read_csv(TAXONOMY_PATH)
    count = len(df)
    assert 2 <= count <= 50, f"Unusual category count: {count}. Expected 2-50 categories."
