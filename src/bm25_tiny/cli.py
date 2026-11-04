import argparse
import json
import os
import re
import sys
from collections.abc import Iterator
from pathlib import Path

import bm25_tiny
from bm25_tiny import BM25, __version__, tokenize

DEFAULT_EXTENSIONS = ".txt,.md,.rst"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bm25-tiny", description=bm25_tiny.__doc__)
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    idx = sub.add_parser("index", help="build an index from text files, folders or JSONL")
    idx.add_argument("paths", nargs="+", type=Path, help="files or folders to index")
    idx.add_argument("-o", "--output", type=Path, required=True, help="index file to write")
    idx.add_argument(
        "--ext",
        default=DEFAULT_EXTENSIONS,
        help=f"comma-separated extensions to pick up in folders (default: {DEFAULT_EXTENSIONS})",
    )
    idx.add_argument("--stopwords", type=Path, help="file with one stopword per line")
    idx.add_argument("--k1", type=float, default=1.5, help="term frequency saturation (1.5)")
    idx.add_argument("--b", type=float, default=0.75, help="length normalization (0.75)")
    idx.add_argument("--delta", type=float, default=0.0, help="BM25+ lower bound (0 = off)")

    search = sub.add_parser("search", help="query an index")
    search.add_argument("index", type=Path, help="index file written by `index`")
    search.add_argument("query", help="search query")
    search.add_argument("-k", type=_positive_int, default=5, help="number of results (5)")
    search.add_argument("--json", action="store_true", help="print results as JSON lines")
    return parser


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def _id_base(paths: list[Path]) -> Path:
    """Folder that document ids are relative to.

    With one folder it is that folder; with several inputs it is their common parent, so
    `a/README.md` and `b/README.md` keep distinct ids.
    """
    if len(paths) == 1 and paths[0].is_dir():
        return paths[0].resolve()
    roots = [str(p.resolve() if p.is_dir() else p.resolve().parent) for p in paths]
    return Path(os.path.commonpath(roots))


def _iter_files(paths: list[Path], extensions: set[str]) -> Iterator[Path]:
    """Yield files to index. Hidden files and folders (.git, .venv) are skipped in folders."""
    for path in paths:
        if path.is_dir():
            for file in sorted(path.rglob("*")):
                hidden = any(part.startswith(".") for part in file.relative_to(path).parts)
                if not hidden and file.is_file() and file.suffix.lower() in extensions:
                    yield file
        elif path.is_file():
            yield path
        else:
            raise FileNotFoundError(f"No such file or folder: {path}")


def _read_jsonl(path: Path) -> Iterator[tuple[str, str, dict]]:
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                doc = json.loads(line)
                doc_id, text, metadata = doc["id"], doc["text"], doc.get("metadata") or {}
            except (json.JSONDecodeError, KeyError, TypeError, AttributeError) as exc:
                raise ValueError(f"{path}:{lineno}: expected {{'id', 'text'}} JSON") from exc
            if not isinstance(text, str) or not isinstance(metadata, dict):
                raise ValueError(f"{path}:{lineno}: 'text' must be a string, 'metadata' an object")
            yield str(doc_id), text, metadata


def _cmd_index(args: argparse.Namespace) -> int:
    stopwords: list[str] = []
    if args.stopwords:
        stopwords = args.stopwords.read_text(encoding="utf-8").split()
    index = BM25(k1=args.k1, b=args.b, delta=args.delta, stopwords=stopwords)
    extensions = {e.strip().lower() for e in args.ext.split(",") if e.strip()}
    extensions = {e if e.startswith(".") else f".{e}" for e in extensions}
    base = _id_base(args.paths)
    skipped = 0
    for file in _iter_files(args.paths, extensions):
        if file.suffix.lower() == ".jsonl":
            for doc_id, text, metadata in _read_jsonl(file):
                index.add(doc_id, text, metadata)
            continue
        try:
            text = file.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            print(f"skipping {file}: not UTF-8 text", file=sys.stderr)
            skipped += 1
            continue
        doc_id = file.resolve().relative_to(base).as_posix()
        index.add(doc_id, text, {"path": str(file)})
    if not len(index):
        print("error: no documents found to index", file=sys.stderr)
        return 1
    index.save(args.output)
    note = f" ({skipped} skipped)" if skipped else ""
    print(f"indexed {len(index)} documents into {args.output}{note}")
    return 0


def snippet(text: str, query: str, width: int = 80) -> str:
    """One line of `text` around the first query term, whitespace collapsed.

    Words are compared with the index tokenizer, so "acao" finds "ação" and "cat" does not
    match inside "concatenate".
    """
    flat = " ".join(text.split())
    terms = set(tokenize(query))
    start = 0
    for word in re.finditer(r"\S+", flat):
        if terms.intersection(tokenize(word.group())):
            start = max(0, word.start() - width // 4)
            break
    piece = flat[start : start + width]
    prefix = "…" if start > 0 else ""
    suffix = "…" if start + width < len(flat) else ""
    return f"{prefix}{piece}{suffix}"


def _cmd_search(args: argparse.Namespace) -> int:
    index = BM25.load(args.index)
    results = index.search(args.query, k=args.k)
    for rank, r in enumerate(results, 1):
        if args.json:
            row = {"rank": rank, "id": r.id, "score": round(r.score, 4), "metadata": r.metadata}
            print(json.dumps(row, ensure_ascii=False))
        else:
            print(f"{rank:>2}. {r.score:7.3f}  {r.id}")
            print(f"    {snippet(r.text, args.query)}")
    if not results and not args.json:
        print("no matches", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "index":
            return _cmd_index(args)
        return _cmd_search(args)
    except (OSError, ValueError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
