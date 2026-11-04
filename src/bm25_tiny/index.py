"""Okapi BM25 inverted index with incremental updates and JSON persistence."""

import copy
import heapq
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bm25_tiny.tokenize import Tokenizer, make_tokenizer

FORMAT_VERSION = 1


def _read_umask() -> int:
    # os.umask can only be read by setting it, and it is process-wide, so do it once at
    # import time rather than on every save (which would race with other threads).
    mask = os.umask(0o022)
    os.umask(mask)
    return mask


_UMASK = _read_umask()


@dataclass(frozen=True)
class Result:
    """One search hit. Scores are only comparable within the same query."""

    id: str
    score: float
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


class BM25:
    """In-memory BM25 index over documents identified by string ids.

    `k1` controls term-frequency saturation and `b` length normalization; the defaults are
    the usual Okapi values. `delta` > 0 turns on BM25+ (Lv & Zhai, 2011), which keeps long
    documents that contain a query term from scoring below short ones that do not.
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        delta: float = 0.0,
        stopwords: Iterable[str] = (),
    ) -> None:
        if not (math.isfinite(k1) and k1 >= 0):
            raise ValueError("k1 must be a finite number >= 0.")
        if not 0 <= b <= 1:
            raise ValueError("b must be between 0 and 1.")
        if not (math.isfinite(delta) and delta >= 0):
            raise ValueError("delta must be a finite number >= 0.")
        if isinstance(stopwords, str):
            raise TypeError("stopwords must be an iterable of words, not a single string.")
        self.k1 = float(k1)
        self.b = float(b)
        self.delta = float(delta)
        self.stopwords = sorted(set(stopwords))
        self._tokenize: Tokenizer = make_tokenizer(self.stopwords)
        self._texts: dict[str, str] = {}
        self._metadata: dict[str, dict[str, Any]] = {}
        self._counts: dict[str, Counter[str]] = {}
        self._lengths: dict[str, int] = {}
        self._postings: dict[str, dict[str, int]] = defaultdict(dict)
        self._total_length = 0

    def __len__(self) -> int:
        return len(self._texts)

    def __contains__(self, doc_id: object) -> bool:
        return doc_id in self._texts

    def add(self, doc_id: str, text: str, metadata: Mapping[str, Any] | None = None) -> None:
        """Index one document. A document with the same id is replaced."""
        if not isinstance(doc_id, str):
            raise TypeError("doc_id must be a string.")
        if not isinstance(text, str):
            raise TypeError("text must be a string.")
        # Tokenize before removing, so a failure never loses the existing document.
        counts = Counter(self._tokenize(text))
        self.remove(doc_id)
        self._texts[doc_id] = text
        self._metadata[doc_id] = dict(metadata or {})
        self._counts[doc_id] = counts
        self._lengths[doc_id] = length = sum(counts.values())
        self._total_length += length
        for term, tf in counts.items():
            self._postings[term][doc_id] = tf

    def add_many(self, docs: Mapping[str, str] | Iterable[tuple[str, str]]) -> None:
        """Index several documents given as a mapping or as (id, text) pairs."""
        items = docs.items() if isinstance(docs, Mapping) else docs
        for doc_id, text in items:
            self.add(doc_id, text)

    def remove(self, doc_id: str) -> bool:
        """Remove a document. Returns False if it was not indexed."""
        counts = self._counts.pop(doc_id, None)
        if counts is None:
            return False
        del self._texts[doc_id]
        del self._metadata[doc_id]
        self._total_length -= self._lengths.pop(doc_id)
        for term in counts:
            postings = self._postings[term]
            del postings[doc_id]
            if not postings:
                del self._postings[term]
        return True

    def get(self, doc_id: str) -> str | None:
        """Return the stored text of a document, or None."""
        return self._texts.get(doc_id)

    def idf(self, word: str) -> float:
        """Inverse document frequency of a single word, normalized like the documents.

        Returns 0 for unseen words, stopwords and input that is not exactly one token.
        """
        tokens = self._tokenize(word)
        return self._idf(tokens[0]) if len(tokens) == 1 else 0.0

    def _idf(self, term: str) -> float:
        df = len(self._postings.get(term, ()))
        if not df:
            return 0.0
        n = len(self._texts)
        # The "+1" variant (as in Lucene) keeps IDF positive for terms in most documents.
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def scores(self, query: str) -> dict[str, float]:
        """Score every document that shares at least one term with `query`."""
        n = len(self._texts)
        if not n:
            return {}
        avg_length = self._total_length / n or 1.0
        result: dict[str, float] = defaultdict(float)
        # A repeated query term counts once per occurrence, as in the original formula.
        for term, qtf in Counter(self._tokenize(query)).items():
            postings = self._postings.get(term)
            if not postings:
                continue
            idf = self._idf(term)
            for doc_id, tf in postings.items():
                norm = 1 - self.b + self.b * self._lengths[doc_id] / avg_length
                tf_part = tf * (self.k1 + 1) / (tf + self.k1 * norm) + self.delta
                result[doc_id] += qtf * idf * tf_part
        return dict(result)

    def search(self, query: str, k: int = 10) -> list[Result]:
        """Return up to `k` best-matching documents, highest score first."""
        if k <= 0:
            return []
        # Ties break on id so the ranking is deterministic.
        best = heapq.nsmallest(k, self.scores(query).items(), key=lambda item: (-item[1], item[0]))
        return [
            Result(
                id=doc_id,
                score=score,
                text=self._texts[doc_id],
                metadata=copy.deepcopy(self._metadata[doc_id]),
            )
            for doc_id, score in best
        ]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to plain JSON-compatible data. Term counts are rebuilt on load."""
        return {
            "format": FORMAT_VERSION,
            "params": {"k1": self.k1, "b": self.b, "delta": self.delta},
            "stopwords": self.stopwords,
            "documents": [
                {"id": doc_id, "text": text, "metadata": self._metadata[doc_id]}
                for doc_id, text in self._texts.items()
            ],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BM25":
        """Rebuild an index produced by `to_dict`."""
        if not isinstance(data, Mapping) or data.get("format") != FORMAT_VERSION:
            found = data.get("format") if isinstance(data, Mapping) else type(data).__name__
            raise ValueError(f"Not a bm25-tiny index (format {found!r}).")
        try:
            params = data["params"]
            index = cls(
                k1=params["k1"],
                b=params["b"],
                delta=params["delta"],
                stopwords=data.get("stopwords", ()),
            )
            for doc in data["documents"]:
                index.add(doc["id"], doc["text"], doc.get("metadata"))
        except (KeyError, TypeError) as exc:
            raise ValueError(f"Malformed bm25-tiny index: {exc!r}.") from exc
        return index

    def save(self, path: str | os.PathLike[str]) -> None:
        """Write the index as JSON atomically, so a crash never leaves a half-written file."""
        target = Path(path)
        fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.")
        try:
            # mkstemp creates the file as 0600; use the usual umask-based mode instead.
            os.chmod(tmp, 0o666 & ~_UMASK)
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(self.to_dict(), fh, ensure_ascii=False)
            os.replace(tmp, target)
        except BaseException:
            os.unlink(tmp)
            raise

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> "BM25":
        """Read an index written by `save`."""
        with open(path, encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))
