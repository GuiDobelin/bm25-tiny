# bm25-tiny

**Okapi BM25 search in pure Python with zero dependencies: index, query, persist.**

[![CI](https://github.com/GuiDobelin/bm25-tiny/actions/workflows/ci.yml/badge.svg)](https://github.com/GuiDobelin/bm25-tiny/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

Keyword search is still the part of retrieval that finds exact error codes, product names and
function names, and it is half of every hybrid RAG pipeline. You should not need
Elasticsearch, NumPy or a C extension for a few thousand documents. `bm25-tiny` is one small,
readable module. You can add and remove documents incrementally, save the index to a JSON
file, and query it from Python or from the command line.

- **Okapi BM25 and BM25+**, with tunable `k1`, `b` and `delta`; the scores are checked against a hand-computed value in the tests
- **Unicode-aware tokenizer**: case folding, accent stripping for Latin/Greek/Cyrillic (`ação` matches `acao`), per-character CJK tokens, combining vowel signs kept in Devanagari, Thai and others
- **Incremental**: `add`, `remove` and replace by id, with no rebuild
- **Persistent**: atomic JSON save/load, with parameters and stopwords stored alongside the documents
- **No runtime dependencies**, Python 3.11+

```text
$ bm25-tiny --help
usage: bm25-tiny [-h] [--version] {index,search} ...

Okapi BM25 search in pure Python with zero dependencies: index, query,
persist.

positional arguments:
  {index,search}
    index         build an index from text files, folders or JSONL
    search        query an index

options:
  -h, --help      show this help message and exit
  --version       show program's version number and exit
```

## Install

```bash
pip install "git+https://github.com/GuiDobelin/bm25-tiny"
```

## Usage

### CLI

Index a folder (`.txt`, `.md` and `.rst` by default; hidden folders such as `.git` and `.venv`
are skipped) and search it, using the small corpus in [`examples/docs`](examples/docs):

```text
$ bm25-tiny index examples/docs -o docs.json
indexed 4 documents into docs.json

$ bm25-tiny search docs.json "exact error codes" -k 3
 1.   2.104  hybrid.md
    …rd matching catches exact terms such as error codes, while embeddings catch para…
 2.   1.963  embeddings.md
    …ry, but it can miss exact identifiers such as error codes, product names or func…

$ bm25-tiny search docs.json "exact error codes" -k 1 --json
{"rank": 1, "id": "hybrid.md", "score": 2.1044, "metadata": {"path": "examples/docs/hybrid.md"}}
```

Snippets are centred on the rarest query term, so a common word like "with" does not pick the
window. Other options:

```bash
bm25-tiny index notes/ papers/ -o i.json          # ids: notes/a.md, papers/b.md
bm25-tiny index docs/ -o i.json --ext md,html      # pick other extensions
bm25-tiny index corpus.jsonl -o i.json             # one {"id", "text", "metadata"?} per line
bm25-tiny index docs/ -o i.json --stopwords stop.txt --k1 1.2 --b 0.75 --delta 1
```

### Python

```python
from bm25_tiny import BM25

index = BM25()  # k1=1.5, b=0.75; BM25(delta=1.0) for BM25+
index.add_many(
    {
        "pasta": "Boil the pasta in salted water for ten minutes.",
        "risotto": "Toast the rice, then add hot stock one ladle at a time.",
        "acai": "Açaí bowl: blend frozen açaí with banana.",
    }
)
index.add("soup", "Carrot and ginger soup with coconut milk.", {"tags": ["vegan"]})

for r in index.search("acai with rice", k=3):
    print(f"{r.id:8} {r.score:.3f}")
```

```text
acai     2.600
risotto  1.032
soup     0.762
```

```python
index.remove("pasta")  # True; incremental, no rebuild
index.save("recipes.json")  # atomic write
index = BM25.load("recipes.json")
index.scores("rice")  # {"risotto": 0.836}: raw scores, e.g. for fusion
```

BM25 scores are only comparable within a single query. To combine them with vector search,
fuse the *rankings*, for example with reciprocal rank fusion, as
[citerag](https://github.com/GuiDobelin/citerag) does.

## Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
ruff check . && ruff format --check .
```

## Ecosystem

Part of a set of small tools around [citerag](https://github.com/GuiDobelin/citerag), a self-hostable
RAG API that answers questions about your documents with verifiable citations.

## License

[MIT](LICENSE)
