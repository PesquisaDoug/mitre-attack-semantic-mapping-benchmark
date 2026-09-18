from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from src.preprocessing import clean_attack_text, normalize_for_audit, mark_strict_query


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(destination.suffix + ".part")
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with tmp.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    tmp.replace(destination)


def ensure_stix_bundle(url: str, destination: Path) -> Path:
    if not destination.exists():
        download_file(url, destination)
    return destination


def load_stix_bundle(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        bundle = json.load(handle)
    if not isinstance(bundle, dict):
        raise TypeError("Expected a STIX JSON object.")
    objects = bundle.get("objects")
    if not isinstance(objects, list):
        raise ValueError("The Enterprise ATT&CK JSON does not contain a valid 'objects' list.")
    return bundle, objects


def is_active_attack_object(obj: dict[str, Any]) -> bool:
    return not bool(obj.get("revoked", False) or obj.get("x_mitre_deprecated", False))


def mitre_external_reference(obj: dict[str, Any]) -> dict[str, Any] | None:
    for ref in obj.get("external_references", []):
        external_id = ref.get("external_id", "")
        if (
            ref.get("source_name") == "mitre-attack"
            and isinstance(external_id, str)
            and external_id.startswith("T")
        ):
            return ref
    return None


def build_technique_corpus(
    objects: list[dict[str, Any]],
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    technique_by_stix_id: dict[str, dict[str, Any]] = {}

    for obj in objects:
        if obj.get("type") != "attack-pattern":
            continue
        if not is_active_attack_object(obj):
            continue

        ref = mitre_external_reference(obj)
        if ref is None:
            continue

        attack_id = ref["external_id"]
        name = clean_attack_text(obj.get("name"))
        description = clean_attack_text(obj.get("description"))
        if not name:
            continue

        tactics = sorted(
            {
                phase.get("phase_name")
                for phase in obj.get("kill_chain_phases", [])
                if phase.get("phase_name")
            }
        )
        platforms = sorted(
            {
                str(platform)
                for platform in obj.get("x_mitre_platforms", [])
                if platform
            }
        )
        document_text = clean_attack_text(f"{attack_id}. {name}. {description}")

        row = {
            "stix_id": obj["id"],
            "attack_id": attack_id,
            "name": name,
            "description": description,
            "document_text": document_text,
            "is_subtechnique": bool(obj.get("x_mitre_is_subtechnique", False)),
            "tactics": json.dumps(tactics),
            "platforms": json.dumps(platforms),
            "url": ref.get("url"),
        }
        rows.append(row)
        technique_by_stix_id[obj["id"]] = row

    techniques = (
        pd.DataFrame(rows)
        .drop_duplicates(subset=["attack_id"])
        .sort_values("attack_id")
        .reset_index(drop=True)
    )
    validate_corpus(techniques)
    return techniques, technique_by_stix_id


def build_source_index(objects: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {obj.get("id"): obj for obj in objects if obj.get("id")}


def source_display_name(source_ref: str, object_by_id: dict[str, dict[str, Any]]) -> str:
    source = object_by_id.get(source_ref, {})
    return clean_attack_text(source.get("name") or source.get("value") or source_ref)


def source_object_type(source_ref: str, object_by_id: dict[str, dict[str, Any]]) -> str:
    source = object_by_id.get(source_ref, {})
    return str(source.get("type", "unknown"))


def source_is_active(source_ref: str, object_by_id: dict[str, dict[str, Any]]) -> bool:
    source = object_by_id.get(source_ref)
    return source is not None and is_active_attack_object(source)


def build_procedure_queries(
    objects: list[dict[str, Any]],
    technique_by_stix_id: dict[str, dict[str, Any]],
    object_by_id: dict[str, dict[str, Any]],
    relationship_type: str = "uses",
) -> pd.DataFrame:
    query_groups: dict[tuple[str, str], dict[str, Any]] = {}

    for obj in objects:
        if obj.get("type") != "relationship":
            continue
        if obj.get("relationship_type") != relationship_type:
            continue
        if not is_active_attack_object(obj):
            continue

        source_ref = obj.get("source_ref")
        target_ref = obj.get("target_ref")
        if not source_ref or not target_ref:
            continue
        if target_ref not in technique_by_stix_id:
            continue
        if not source_is_active(source_ref, object_by_id):
            continue

        description = clean_attack_text(obj.get("description"))
        if not description:
            continue

        source_name = source_display_name(source_ref, object_by_id)
        query_text = clean_attack_text(f"{source_name}. {description}")
        if len(query_text) < 20:
            continue

        group_key = (source_ref, normalize_for_audit(description))
        if group_key not in query_groups:
            query_groups[group_key] = {
                "source_ref": source_ref,
                "source_name": source_name,
                "source_type": source_object_type(source_ref, object_by_id),
                "description": description,
                "query_text": query_text,
                "relationship_ids": [],
                "relevant_attack_ids": set(),
                "relevant_names": set(),
            }

        query_groups[group_key]["relationship_ids"].append(obj.get("id"))
        target = technique_by_stix_id[target_ref]
        query_groups[group_key]["relevant_attack_ids"].add(target["attack_id"])
        query_groups[group_key]["relevant_names"].add(target["name"])

    rows: list[dict[str, Any]] = []
    sorted_groups = sorted(
        query_groups.values(),
        key=lambda item: (item["source_type"], item["source_name"], item["query_text"]),
    )
    for index, group in enumerate(sorted_groups):
        relevant_ids = sorted(group["relevant_attack_ids"])
        relevant_names = sorted(group["relevant_names"])
        contains_target_id, contains_exact_target_name, strict_eligible = mark_strict_query(
            group["query_text"],
            relevant_ids,
            relevant_names,
        )
        rows.append(
            {
                "query_id": f"Q{index + 1:06d}",
                "source_ref": group["source_ref"],
                "source_name": group["source_name"],
                "source_type": group["source_type"],
                "description": group["description"],
                "query_text": group["query_text"],
                "relevant_attack_ids": json.dumps(relevant_ids),
                "relevant_names": json.dumps(relevant_names),
                "relevant_count": len(relevant_ids),
                "relationship_ids": json.dumps(
                    sorted(value for value in group["relationship_ids"] if value)
                ),
                "contains_target_id": contains_target_id,
                "contains_exact_target_name": contains_exact_target_name,
                "strict_eligible": strict_eligible,
            }
        )

    queries = pd.DataFrame(rows)
    validate_queries(queries)
    return queries


def sample_queries(queries: pd.DataFrame, max_queries: int | None, seed: int) -> pd.DataFrame:
    if max_queries is not None and len(queries) > max_queries:
        return (
            queries.sample(n=max_queries, random_state=seed)
            .sort_values("query_id")
            .reset_index(drop=True)
        )
    return queries.copy().reset_index(drop=True)


def validate_corpus(techniques: pd.DataFrame) -> None:
    if techniques.empty:
        raise RuntimeError("No Enterprise ATT&CK techniques were extracted.")
    if techniques["attack_id"].duplicated().any():
        raise RuntimeError("Duplicate ATT&CK IDs remain in the technique corpus.")


def validate_queries(queries: pd.DataFrame) -> None:
    if queries.empty:
        raise RuntimeError("No ATT&CK procedure-description queries were constructed.")
    if not (queries["relevant_count"] >= 1).all():
        raise RuntimeError("Every query must have at least one relevant ATT&CK technique.")


def data_quality_summary(techniques: pd.DataFrame, all_queries: pd.DataFrame, queries: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"metric": "candidate_techniques", "value": len(techniques)},
            {"metric": "all_procedure_queries_available", "value": len(all_queries)},
            {"metric": "evaluation_queries", "value": len(queries)},
            {"metric": "strict_queries", "value": int(queries["strict_eligible"].sum())},
            {"metric": "target_id_leakage_queries", "value": int(queries["contains_target_id"].sum())},
            {
                "metric": "exact_target_name_leakage_queries",
                "value": int(queries["contains_exact_target_name"].sum()),
            },
            {"metric": "multi_relevant_queries", "value": int((queries["relevant_count"] > 1).sum())},
        ]
    )


def query_source_type_counts(queries: pd.DataFrame) -> pd.DataFrame:
    return queries["source_type"].value_counts().rename_axis("source_type").reset_index(name="queries")


def create_data_manifest(
    objects: list[dict[str, Any]],
    bundle_path: Path,
    bundle_url: str,
    techniques: pd.DataFrame,
    all_queries: pd.DataFrame,
    queries: pd.DataFrame,
    canonical_repository: str,
    domain: str,
) -> dict[str, Any]:
    modified_values = [obj.get("modified") for obj in objects if obj.get("modified")]
    return {
        "dataset_name": "MITRE ATT&CK Enterprise STIX 2.1",
        "domain": domain,
        "canonical_repository": canonical_repository,
        "bundle_url": bundle_url,
        "bundle_file": bundle_path.name,
        "bundle_sha256": sha256_file(bundle_path),
        "bundle_size_bytes": bundle_path.stat().st_size,
        "stix_object_count": len(objects),
        "latest_object_modified": max(modified_values) if modified_values else None,
        "candidate_technique_count": len(techniques),
        "all_constructed_query_count": len(all_queries),
        "evaluation_query_count": len(queries),
        "strict_query_count": int(queries["strict_eligible"].sum()),
        "query_construction": (
            "Active STIX 'uses' relationships with non-empty descriptions; "
            "queries are grouped by source_ref + normalized description."
        ),
        "strict_subset_rule": (
            "Exclude query if exact relevant ATT&CK ID or exact relevant technique name "
            "occurs in normalized query text."
        ),
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
