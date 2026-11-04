"""Okapi BM25 search in pure Python with zero dependencies: index, query, persist."""

from bm25_tiny.index import BM25, Result
from bm25_tiny.tokenize import make_tokenizer, tokenize

__version__ = "0.1.0"

__all__ = ["BM25", "Result", "__version__", "make_tokenizer", "tokenize"]
