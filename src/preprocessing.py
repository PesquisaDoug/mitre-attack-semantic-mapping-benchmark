from __future__ import annotations

import html
import re
from collections.abc import Iterable

CITATION_PATTERN = re.compile(r"\(Citation:[^)]+\)", flags=re.IGNORECASE)
MARKDOWN_LINK_PATTERN = re.compile(r"\[([^\]]+)\]\([^)]+\)")
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
MULTISPACE_PATTERN = re.compile(r"\s+")


def clean_attack_text(text: str | None) -> str:
    if text is None:
        return ""

    value = html.unescape(str(text))
    value = MARKDOWN_LINK_PATTERN.sub(r"\1", value)
    value = CITATION_PATTERN.sub(" ", value)
    value = HTML_TAG_PATTERN.sub(" ", value)
    value = value.replace("\u00a0", " ")
    value = MULTISPACE_PATTERN.sub(" ", value)
    return value.strip()


def normalize_for_audit(text: str) -> str:
    value = clean_attack_text(text).lower()
    value = re.sub(r"[^a-z0-9.]+", " ", value)
    value = MULTISPACE_PATTERN.sub(" ", value)
    return value.strip()


def simple_tokenize(text: str) -> list[str]:
    return re.findall(
        r"[a-z0-9][a-z0-9._/-]*",
        clean_attack_text(text).lower(),
    )


def detect_target_id_leakage(query_text: str, relevant_attack_ids: Iterable[str]) -> bool:
    normalized_query = normalize_for_audit(query_text)
    return any(
        normalize_for_audit(attack_id) in normalized_query
        for attack_id in relevant_attack_ids
    )


def detect_exact_target_name_leakage(query_text: str, relevant_names: Iterable[str]) -> bool:
    normalized_query = normalize_for_audit(query_text)
    return any(
        normalized_name in normalized_query
        for normalized_name in (normalize_for_audit(name) for name in relevant_names)
        if len(normalized_name) >= 4
    )


def mark_strict_query(
    query_text: str,
    relevant_attack_ids: Iterable[str],
    relevant_names: Iterable[str],
) -> tuple[bool, bool, bool]:
    contains_target_id = detect_target_id_leakage(query_text, relevant_attack_ids)
    contains_exact_target_name = detect_exact_target_name_leakage(query_text, relevant_names)
    strict_eligible = not (contains_target_id or contains_exact_target_name)
    return contains_target_id, contains_exact_target_name, strict_eligible
