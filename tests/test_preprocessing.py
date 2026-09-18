from src.preprocessing import clean_attack_text, mark_strict_query, normalize_for_audit, simple_tokenize


def test_clean_attack_text_preserves_markdown_anchor_and_removes_noise():
    text = 'Use [PowerShell](https://example.test) <b>payload</b> (Citation: X) &amp; tools'
    assert clean_attack_text(text) == "Use PowerShell payload & tools"


def test_normalize_and_tokenize():
    assert normalize_for_audit("T1059.001: PowerShell!") == "t1059.001 powershell"
    assert simple_tokenize("PowerShell.exe /c test") == ["powershell.exe", "c", "test"]


def test_strict_flags():
    contains_id, contains_name, strict = mark_strict_query(
        "Actor used T1059.001 PowerShell",
        ["T1059.001"],
        ["PowerShell"],
    )
    assert contains_id is True
    assert contains_name is True
    assert strict is False


def test_strict_without_exact_target():
    contains_id, contains_name, strict = mark_strict_query(
        "Actor launched commands through a scripting shell",
        ["T1059.001"],
        ["PowerShell"],
    )
    assert contains_id is False
    assert contains_name is False
    assert strict is True
