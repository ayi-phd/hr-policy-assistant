"""Policy loading and deterministic keyword search.

No embeddings, no vector DB - just tokenised term-frequency scoring with a
title-match boost. This is the domain layer the ``search_policies`` tool wraps.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from app.models import Policy, PolicySearchHit

DEFAULT_SEARCH_LIMIT = 3
_TITLE_BOOST = 3.0

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Small, generic stopword set - enough to stop trivial words from dominating
# the score without needing a linguistics library.
_STOPWORDS = frozenset(
    """
    a an and are as at be but by can could do does for from had has have how i if in
    into is it its may me my no not of on or our shall should so than that the their
    them then there these they this to up us was we were what when where which who will
    with would you your
    """.split()
)


def _tokenize(text: str) -> list[str]:
    return [tok for tok in _TOKEN_RE.findall(text.lower()) if tok not in _STOPWORDS]


def _extract_title(text: str, fallback: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()
    return fallback


def load_policies(policy_dir: Path) -> list[Policy]:
    """Load every ``*.md`` file in *policy_dir* into a :class:`Policy`.

    Title comes from the first level-1 heading, falling back to the file stem.
    Results are sorted by source filename for deterministic ordering.
    """

    if not policy_dir.is_dir():
        raise FileNotFoundError(f"policy directory not found: {policy_dir}")

    policies: list[Policy] = []
    for path in sorted(policy_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        policies.append(
            Policy(
                title=_extract_title(text, path.stem),
                source=path.name,
                content=text,
            )
        )
    return policies


def _score(query_tokens: list[str], policy: Policy) -> float:
    if not query_tokens:
        return 0.0

    body_counts = Counter(_tokenize(policy.content))
    title_tokens = set(_tokenize(policy.title))

    score = 0.0
    for token in set(query_tokens):
        score += float(body_counts.get(token, 0))
        if token in title_tokens:
            score += _TITLE_BOOST
    return score


def search_policies(
    policies: list[Policy],
    query: str,
    limit: int = DEFAULT_SEARCH_LIMIT,
) -> list[PolicySearchHit]:
    """Return up to *limit* policies ranked by keyword relevance to *query*.

    An empty or whitespace-only query returns an empty list.
    """

    query_tokens = _tokenize(query or "")
    if not query_tokens:
        return []

    scored = (
        PolicySearchHit(
            title=p.title,
            source=p.source,
            content=p.content,
            score=_score(query_tokens, p),
        )
        for p in policies
    )
    hits = [hit for hit in scored if hit.score > 0]
    hits.sort(key=lambda h: (-h.score, h.source))
    return hits[: max(limit, 0)]


class PolicyStore:
    """In-memory collection of loaded policies with a search method."""

    def __init__(self, policies: list[Policy]) -> None:
        self._policies = list(policies)

    @classmethod
    def from_directory(cls, policy_dir: Path) -> "PolicyStore":
        return cls(load_policies(policy_dir))

    @property
    def policies(self) -> list[Policy]:
        return list(self._policies)

    def __len__(self) -> int:
        return len(self._policies)

    def search(self, query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[PolicySearchHit]:
        return search_policies(self._policies, query, limit)
