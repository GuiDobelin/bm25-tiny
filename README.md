# bm25-tiny

**Okapi BM25 search in pure Python with zero dependencies: index, query, persist**

[![CI](https://github.com/GuiDobelin/bm25-tiny/actions/workflows/ci.yml/badge.svg)](https://github.com/GuiDobelin/bm25-tiny/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

<!-- TODO: one paragraph on the problem this solves and who it is for. -->

```text
$ bm25-tiny --help
```

## Install

```bash
pip install "git+https://github.com/GuiDobelin/bm25-tiny"
```

## Usage

### CLI

```bash
bm25-tiny ...
```

### Python

```python
import bm25_tiny
```

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
