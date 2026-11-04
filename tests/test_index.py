import json
import math

import pytest

from bm25_tiny import BM25, Result

DOCS = {
    "fox": "The quick brown fox jumps over the lazy dog",
    "dog": "A lazy dog sleeps all day",
    "cat": "Cats are independent and curious animals",
}


@pytest.fixture
def index():
    idx = BM25()
    idx.add_many(DOCS)
    return idx


def test_ranks_matching_documents(index):
    results = index.search("lazy dog")
    assert [r.id for r in results] == ["dog", "fox"]
    assert results[0].score > results[1].score > 0
    assert results[0].text == DOCS["dog"]


def test_matches_hand_computed_score():
    idx = BM25(k1=1.2, b=0.75)
    idx.add_many({"a": "apple banana", "b": "banana banana cherry date"})
    # n=2, avgdl=3, df(banana)=2; doc b: tf=2, dl=4.
    idf = math.log(1 + (2 - 2 + 0.5) / (2 + 0.5))
    norm = 1 - 0.75 + 0.75 * 4 / 3
    expected = idf * 2 * 2.2 / (2 + 1.2 * norm)
    assert idx.scores("banana")["b"] == pytest.approx(expected)


def test_rare_terms_weigh_more(index):
    assert index.idf("cats") > index.idf("lazy") > 0
    assert index.idf("missing") == 0


def test_no_match_and_empty_index():
    assert BM25().search("anything") == []
    idx = BM25()
    idx.add("a", "hello world")
    assert idx.search("goodbye") == []
    assert idx.search("") == []


def test_k_limits_results(index):
    assert len(index.search("lazy dog", k=1)) == 1
    assert index.search("lazy dog", k=0) == []
    assert index.search("lazy dog", k=-3) == []


def test_ties_break_on_id():
    idx = BM25()
    idx.add_many([("b", "same text"), ("a", "same text")])
    assert [r.id for r in idx.search("text")] == ["a", "b"]


def test_repeated_query_terms_count_per_occurrence(index):
    once = index.scores("dog")["dog"]
    assert index.scores("dog dog")["dog"] == pytest.approx(2 * once)


def test_add_replaces_existing_document(index):
    index.add("dog", "completely different content")
    assert len(index) == 3
    assert "dog" not in [r.id for r in index.search("sleeps")]
    assert [r.id for r in index.search("different")] == ["dog"]


def test_remove(index):
    assert index.remove("dog") is True
    assert index.remove("dog") is False
    assert "dog" not in index
    assert len(index) == 2
    assert [r.id for r in index.search("lazy")] == ["fox"]
    assert index.idf("sleeps") == 0


def test_remove_restores_scores_of_fresh_index(index):
    index.add("extra", "lazy lazy lazy words here")
    index.remove("extra")
    fresh = BM25()
    fresh.add_many(DOCS)
    assert index.scores("lazy dog") == pytest.approx(fresh.scores("lazy dog"))


def test_metadata_is_returned(index):
    index.add("meta", "unique zebra", {"path": "z.txt"})
    assert index.search("zebra") == [
        Result(
            id="meta",
            score=index.scores("zebra")["meta"],
            text="unique zebra",
            metadata={"path": "z.txt"},
        )
    ]


def test_stopwords_are_ignored():
    idx = BM25(stopwords=["the"])
    idx.add_many({"a": "the the the", "b": "the cat"})
    assert idx.search("the") == []
    assert [r.id for r in idx.search("the cat")] == ["b"]


def test_bm25_plus_helps_long_matching_documents():
    long_doc = "rare " + "filler " * 200
    docs = {"long": long_doc, "short": "short words only", "other": "other words"}
    plain, plus = BM25(), BM25(delta=1.0)
    plain.add_many(docs)
    plus.add_many(docs)
    assert plus.scores("rare")["long"] > plain.scores("rare")["long"]


def test_k1_zero_ignores_term_frequency():
    idx = BM25(k1=0)
    idx.add_many({"a": "x", "b": "x x x", "c": "y"})
    s = idx.scores("x")
    assert s["a"] == pytest.approx(s["b"])


def test_iterates_over_ids(index):
    assert list(index) == ["fox", "dog", "cat"]


def test_get(index):
    assert index.get("cat") == DOCS["cat"]
    assert index.get("nope") is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"k1": -1},
        {"k1": float("inf")},
        {"k1": float("nan")},
        {"b": 1.5},
        {"b": -0.1},
        {"b": float("nan")},
        {"delta": -1},
        {"delta": float("nan")},
    ],
)
def test_rejects_invalid_params(kwargs):
    with pytest.raises(ValueError):
        BM25(**kwargs)


def test_rejects_non_string_ids():
    with pytest.raises(TypeError):
        BM25().add(1, "text")  # type: ignore[arg-type]


def test_save_and_load_roundtrip(tmp_path, index):
    index.add("meta", "ação", {"lang": "pt"})
    path = tmp_path / "index.json"
    index.save(path)
    loaded = BM25.load(path)
    assert len(loaded) == len(index)
    assert loaded.search("acao") == index.search("acao")
    assert loaded.scores("lazy dog") == index.scores("lazy dog")
    assert not list(tmp_path.glob(".index.json.*"))


def test_save_keeps_params_and_stopwords(tmp_path):
    idx = BM25(k1=1.2, b=0.5, delta=0.5, stopwords=["the"])
    idx.add("a", "the cat")
    idx.save(tmp_path / "i.json")
    loaded = BM25.load(tmp_path / "i.json")
    assert (loaded.k1, loaded.b, loaded.delta, loaded.stopwords) == (1.2, 0.5, 0.5, ["the"])
    assert loaded.search("the") == []


def test_save_failure_leaves_no_temp_file(tmp_path, index):
    index.add("bad", "x", {"obj": object()})
    with pytest.raises(TypeError):
        index.save(tmp_path / "index.json")
    assert list(tmp_path.iterdir()) == []


def test_load_rejects_unknown_format(tmp_path):
    path = tmp_path / "i.json"
    path.write_text(json.dumps({"format": 99}))
    with pytest.raises(ValueError, match="format"):
        BM25.load(path)


def test_failed_add_keeps_existing_document(index):
    with pytest.raises(TypeError):
        index.add("dog", None)  # type: ignore[arg-type]
    assert index.get("dog") == DOCS["dog"]
    assert [r.id for r in index.search("sleeps")] == ["dog"]


def test_result_metadata_is_a_copy(index):
    index.add("m", "zebra", {"tags": ["a"]})
    index.search("zebra")[0].metadata["tags"].append("b")
    assert index.search("zebra")[0].metadata == {"tags": ["a"]}


def test_idf_normalizes_the_word(index):
    assert index.idf("Cats") == index.idf("cats") > 0
    assert index.idf("lazy dog") == 0


def test_stopwords_reject_single_string():
    with pytest.raises(TypeError):
        BM25(stopwords="the")


def test_saved_file_follows_umask(tmp_path, index, monkeypatch):
    monkeypatch.setattr("bm25_tiny.index._UMASK", 0o022)
    index.save(tmp_path / "i.json")
    assert (tmp_path / "i.json").stat().st_mode & 0o777 == 0o644


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {"format": 1},
        {"format": 1, "params": {"k1": 1.5}, "documents": []},
        {"format": 1, "params": {"k1": 1.5, "b": 0.75, "delta": 0}, "documents": [{"id": "a"}]},
    ],
)
def test_load_rejects_malformed_files(tmp_path, payload):
    path = tmp_path / "i.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="bm25-tiny index"):
        BM25.load(path)


def test_search_matches_full_sort_for_any_k():
    idx = BM25()
    idx.add_many({f"d{i}": "common " * (i % 5 + 1) + f"w{i}" for i in range(50)})
    full = idx.search("common", k=100)
    for k in (1, 7, 50):
        assert idx.search("common", k=k) == full[:k]
