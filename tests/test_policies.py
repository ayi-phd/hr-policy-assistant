"""Tests for policy loading and keyword search."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.policies import PolicyStore, load_policies, search_policies


def test_load_policies_reads_only_markdown_and_parses_title(policy_dir: Path) -> None:
    policies = load_policies(policy_dir)

    assert {p.source for p in policies} == {
        "Business Travel Policy.md",
        "Paid Time Off Policy.md",
        "Remote Work Policy.md",
    }
    remote = next(p for p in policies if p.source == "Remote Work Policy.md")
    assert remote.title == "Remote Work Policy"
    assert "10 business days" in remote.content


def test_load_policies_missing_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_policies(tmp_path / "does-not-exist")


def test_title_falls_back_to_filename_stem(tmp_path: Path) -> None:
    (tmp_path / "Untitled Policy.md").write_text("no heading here\n", encoding="utf-8")
    (policies := load_policies(tmp_path))
    assert policies[0].title == "Untitled Policy"


def test_search_returns_relevant_policy_first(policy_store: PolicyStore) -> None:
    hits = policy_store.search("work remotely from another state")

    assert hits, "expected at least one hit"
    assert hits[0].title == "Remote Work Policy"
    assert hits[0].score > 0


def test_search_excludes_irrelevant_policies(policy_store: PolicyStore) -> None:
    hits = policy_store.search("vacation accrual carryover days")
    titles = [h.title for h in hits]

    assert "Paid Time Off Policy" in titles
    assert "Business Travel Policy" not in titles


def test_empty_or_whitespace_query_returns_nothing(policy_store: PolicyStore) -> None:
    assert policy_store.search("") == []
    assert policy_store.search("   ") == []
    # A query made entirely of stopwords also yields nothing.
    assert policy_store.search("the and of to") == []


def test_search_respects_limit(policy_store: PolicyStore) -> None:
    hits = policy_store.search("approval policy employees travel work", limit=2)
    assert len(hits) <= 2


def test_search_orders_by_score_then_source() -> None:
    from app.models import Policy

    policies = [
        Policy(title="Alpha", source="b.md", content="remote remote remote"),
        Policy(title="Beta", source="a.md", content="remote remote remote"),
    ]
    hits = search_policies(policies, "remote")
    # Equal score -> tie-broken by source filename ascending.
    assert [h.source for h in hits] == ["a.md", "b.md"]
