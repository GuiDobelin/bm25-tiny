import json

import pytest

from bm25_tiny import BM25, __version__
from bm25_tiny.cli import main, snippet


@pytest.fixture
def corpus(tmp_path):
    docs = tmp_path / "docs"
    (docs / "sub").mkdir(parents=True)
    (docs / "fox.md").write_text("The quick brown fox jumps over the lazy dog.", encoding="utf-8")
    (docs / "sub" / "dog.txt").write_text("A lazy dog sleeps all day long.", encoding="utf-8")
    (docs / "ignored.py").write_text("print('lazy dog')", encoding="utf-8")
    return docs


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_requires_a_command():
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code == 2


def test_index_and_search(tmp_path, corpus, capsys):
    out = tmp_path / "index.json"
    assert main(["index", str(corpus), "-o", str(out)]) == 0
    assert "indexed 2 documents" in capsys.readouterr().out
    assert sorted(BM25.load(out)) == ["fox.md", "sub/dog.txt"]

    assert main(["search", str(out), "lazy dog sleeps"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].endswith("sub/dog.txt")
    assert "sleeps" in lines[1]
    assert lines[2].endswith("fox.md")


def test_search_json(tmp_path, corpus, capsys):
    out = tmp_path / "index.json"
    main(["index", str(corpus), "-o", str(out)])
    capsys.readouterr()
    assert main(["search", str(out), "fox", "--json", "-k", "1"]) == 0
    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert len(rows) == 1
    assert rows[0]["id"] == "fox.md" and rows[0]["rank"] == 1
    assert rows[0]["metadata"]["path"].endswith("fox.md")


def test_ext_option(tmp_path, corpus, capsys):
    out = tmp_path / "index.json"
    assert main(["index", str(corpus), "-o", str(out), "--ext", "py"]) == 0
    assert list(BM25.load(out)) == ["ignored.py"]


def test_index_jsonl(tmp_path, capsys):
    src = tmp_path / "docs.jsonl"
    src.write_text(
        '{"id": "a", "text": "alpha beta"}\n\n{"id": 2, "text": "gamma", "metadata": {"x": 1}}\n',
        encoding="utf-8",
    )
    out = tmp_path / "i.json"
    assert main(["index", str(src), "-o", str(out)]) == 0
    index = BM25.load(out)
    assert index.search("gamma")[0].id == "2"
    assert index.search("gamma")[0].metadata == {"x": 1}


def test_index_jsonl_bad_line(tmp_path, capsys):
    src = tmp_path / "docs.jsonl"
    src.write_text('{"id": "a"}\n', encoding="utf-8")
    assert main(["index", str(src), "-o", str(tmp_path / "i.json")]) == 1
    assert "docs.jsonl:1" in capsys.readouterr().err


def test_stopwords_and_params(tmp_path, corpus, capsys):
    stop = tmp_path / "stop.txt"
    stop.write_text("the\nover\n", encoding="utf-8")
    out = tmp_path / "i.json"
    args = ["index", str(corpus), "-o", str(out), "--stopwords", str(stop), "--k1", "1.2"]
    assert main(args) == 0
    index = BM25.load(out)
    assert index.k1 == 1.2
    assert index.search("the") == []


def test_skips_non_utf8_files(tmp_path, corpus, capsys):
    (corpus / "binary.txt").write_bytes(b"\xff\xfe\x00bad")
    out = tmp_path / "i.json"
    assert main(["index", str(corpus), "-o", str(out)]) == 0
    captured = capsys.readouterr()
    assert "1 skipped" in captured.out
    assert "binary.txt" in captured.err


def test_missing_path(tmp_path, capsys):
    assert main(["index", str(tmp_path / "nope"), "-o", str(tmp_path / "i.json")]) == 1
    assert "No such file" in capsys.readouterr().err


def test_empty_folder(tmp_path, capsys):
    assert main(["index", str(tmp_path), "-o", str(tmp_path / "i.json")]) == 1
    assert "no documents" in capsys.readouterr().err
    assert not (tmp_path / "i.json").exists()


def test_invalid_params(tmp_path, corpus, capsys):
    assert main(["index", str(corpus), "-o", str(tmp_path / "i.json"), "--b", "2"]) == 1
    assert "b must be" in capsys.readouterr().err


def test_search_missing_index(tmp_path, capsys):
    assert main(["search", str(tmp_path / "nope.json"), "x"]) == 1


def test_search_invalid_k(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        main(["search", str(tmp_path / "i.json"), "x", "-k", "0"])
    assert exc.value.code == 2
    assert "at least 1" in capsys.readouterr().err


def test_same_name_in_two_inputs_keeps_both(tmp_path, capsys):
    for name in ("a", "b"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "README.md").write_text(f"readme {name}", encoding="utf-8")
    out = tmp_path / "i.json"
    assert main(["index", str(tmp_path / "a"), str(tmp_path / "b"), "-o", str(out)]) == 0
    assert sorted(BM25.load(out)) == ["a/README.md", "b/README.md"]


def test_skips_hidden_folders(tmp_path, corpus, capsys):
    (corpus / ".venv").mkdir()
    (corpus / ".venv" / "LICENSE.txt").write_text("lazy dog license", encoding="utf-8")
    (corpus / ".notes.md").write_text("lazy", encoding="utf-8")
    out = tmp_path / "i.json"
    assert main(["index", str(corpus), "-o", str(out)]) == 0
    assert sorted(BM25.load(out)) == ["fox.md", "sub/dog.txt"]


def test_explicit_hidden_file_is_indexed(tmp_path, capsys):
    (tmp_path / ".notes.md").write_text("secret plan", encoding="utf-8")
    out = tmp_path / "i.json"
    assert main(["index", str(tmp_path / ".notes.md"), "-o", str(out)]) == 0
    assert list(BM25.load(out)) == [".notes.md"]


@pytest.mark.parametrize(
    "line", ['{"id": "a", "text": 5}', '{"id": "a", "text": "x", "metadata": [1]}', "[1, 2]"]
)
def test_index_jsonl_bad_types(tmp_path, capsys, line):
    src = tmp_path / "docs.jsonl"
    src.write_text(line + "\n", encoding="utf-8")
    assert main(["index", str(src), "-o", str(tmp_path / "i.json")]) == 1
    assert "docs.jsonl:1" in capsys.readouterr().err


def test_search_no_matches(tmp_path, corpus, capsys):
    out = tmp_path / "i.json"
    main(["index", str(corpus), "-o", str(out)])
    capsys.readouterr()
    assert main(["search", str(out), "zebra"]) == 0
    captured = capsys.readouterr()
    assert captured.out == "" and "no matches" in captured.err


def test_snippet_centers_on_match():
    text = "word " * 50 + "needle here " + "tail " * 50
    s = snippet(text, "Needle", width=40)
    assert "needle" in s and s.startswith("…") and s.endswith("…")


def test_snippet_short_text_and_no_match():
    assert snippet("short   text\nhere", "zzz") == "short text here"


def test_snippet_matches_accent_folded_words():
    text = "intro " * 40 + "a ação começa " + "fim " * 40
    assert "ação" in snippet(text, "acao", width=30)


def test_snippet_matches_whole_words_only():
    text = "concatenate " * 20 + "the cat sat"
    assert "the cat" in snippet(text, "cat", width=30)
