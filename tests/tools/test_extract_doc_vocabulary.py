import pytest

from qwen_agent.tools import ExtractDocVocabulary


def test_extract_doc_vocabulary_cache_roundtrip(tmp_path, monkeypatch):
    """Regression test: the cached vocabulary must be identical to the freshly
    computed one. The cache is written with json.dumps but read back verbatim
    by Storage.get, so the second call returned an extra pair of quotes."""
    pytest.importorskip("sklearn.feature_extraction.text")

    tool = ExtractDocVocabulary(cfg={"path": str(tmp_path)})

    def fake_parse(params, **kwargs):
        return "alpha beta gamma alpha beta"

    monkeypatch.setattr(tool, "simple_doc_parse", type("Stub", (), {"call": staticmethod(fake_parse)})())

    files = ["fake_a.txt", "fake_b.txt"]
    first = tool.call({"files": files})
    second = tool.call({"files": files})

    assert second == first
    assert not (second.startswith('"') and second.endswith('"'))
