import json
from pathlib import Path

from src.data import (
    build_procedure_queries,
    build_source_index,
    build_technique_corpus,
    is_active_attack_object,
    mitre_external_reference,
    sha256_file,
)


def test_active_filter_and_external_reference():
    active = {"type": "attack-pattern"}
    revoked = {"type": "attack-pattern", "revoked": True}
    deprecated = {"type": "attack-pattern", "x_mitre_deprecated": True}
    assert is_active_attack_object(active)
    assert not is_active_attack_object(revoked)
    assert not is_active_attack_object(deprecated)
    ref = mitre_external_reference(
        {"external_references": [{"source_name": "mitre-attack", "external_id": "T1001"}]}
    )
    assert ref["external_id"] == "T1001"


def test_corpus_and_multi_relevance_query_construction():
    objects = [
        {
            "type": "attack-pattern",
            "id": "attack-pattern--1",
            "name": "PowerShell",
            "description": "Command execution",
            "external_references": [{"source_name": "mitre-attack", "external_id": "T1059.001"}],
        },
        {
            "type": "attack-pattern",
            "id": "attack-pattern--2",
            "name": "Ingress Tool Transfer",
            "description": "Transfer tools",
            "external_references": [{"source_name": "mitre-attack", "external_id": "T1105"}],
        },
        {"type": "intrusion-set", "id": "intrusion-set--1", "name": "APT Test"},
        {
            "type": "relationship",
            "id": "relationship--1",
            "relationship_type": "uses",
            "source_ref": "intrusion-set--1",
            "target_ref": "attack-pattern--1",
            "description": "Downloads a payload then executes commands.",
        },
        {
            "type": "relationship",
            "id": "relationship--2",
            "relationship_type": "uses",
            "source_ref": "intrusion-set--1",
            "target_ref": "attack-pattern--2",
            "description": "Downloads a payload then executes commands.",
        },
    ]
    techniques, by_stix = build_technique_corpus(objects)
    queries = build_procedure_queries(objects, by_stix, build_source_index(objects))
    assert techniques["attack_id"].tolist() == ["T1059.001", "T1105"]
    assert len(queries) == 1
    assert json.loads(queries.iloc[0]["relevant_attack_ids"]) == ["T1059.001", "T1105"]
    assert queries.iloc[0]["relevant_count"] == 2


def test_sha256_file(tmp_path: Path):
    path = tmp_path / "x.txt"
    path.write_text("abc", encoding="utf-8")
    assert sha256_file(path) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
