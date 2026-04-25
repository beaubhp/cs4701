from __future__ import annotations

import re


TOKEN_RE = re.compile(r"\d+(?:\.\d+)+|[a-z0-9]+(?:['-][a-z0-9]+)*")

STOPWORDS = {
    "a",
    "about",
    "after",
    "all",
    "also",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "do",
    "does",
    "for",
    "from",
    "how",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "on",
    "or",
    "that",
    "the",
    "their",
    "there",
    "these",
    "this",
    "to",
    "under",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "whom",
    "with",
}

QUERY_EXPANSIONS = {
    "leave of absence": ["loa", "hloa", "ploa"],
    "title iv": ["withdrawal", "enrollment", "status"],
    "academic integrity": ["unauthorized", "assistance", "plagiarism"],
    "discipline": ["sanction", "remedy"],
}


def normalize_text(text: str) -> str:
    text = text.replace("’", "'").replace("‘", "'")
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"\s+-\s+", "-", text)
    return text.lower()


def tokenize(text: str, remove_stopwords: bool = True) -> list[str]:
    normalized = normalize_text(text)
    tokens = []
    for match in TOKEN_RE.finditer(normalized):
        token = strip_possessive(match.group(0))
        if not token:
            continue
        if remove_stopwords and token in STOPWORDS:
            continue
        tokens.append(token)
    return tokens


def strip_possessive(token: str) -> str:
    if token.endswith("'s") and len(token) > 2:
        return token[:-2]
    return token


def expand_query_text(query: str) -> str:
    normalized = normalize_text(query)
    expansions: list[str] = []
    for phrase, terms in QUERY_EXPANSIONS.items():
        if phrase in normalized:
            expansions.extend(terms)
    if not expansions:
        return query
    return f"{query} {' '.join(expansions)}"
