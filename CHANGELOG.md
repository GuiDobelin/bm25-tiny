# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.0] - 2026-11-04

### Added

- `BM25` index with Okapi BM25 and BM25+ scoring, incremental `add`/`remove`, metadata,
  and atomic JSON `save`/`load`.
- Unicode-aware `tokenize` (case folding, accent stripping for Latin/Greek/Cyrillic, CJK
  per character, combining vowel signs kept) and `make_tokenizer` with stopwords.
- `bm25-tiny index` for files, folders and JSONL, and `bm25-tiny search` with snippets and
  `--json` output.
