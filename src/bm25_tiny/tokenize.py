"""Unicode-aware tokenizer: case folding, accent stripping and CJK splitting."""

import unicodedata
from collections.abc import Callable, Iterable

Tokenizer = Callable[[str], list[str]]

# Accents are only stripped from Latin, Greek and Cyrillic letters; in scripts such as
# Devanagari or Thai the combining marks are vowels and change the word.
_FOLDABLE_BELOW = 0x0530


def _is_cjk(ch: str) -> bool:
    # Chinese and Japanese are written without spaces, so each ideograph or kana is a token.
    cp = ord(ch)
    return (
        0x3040 <= cp <= 0x30FF
        or 0x3400 <= cp <= 0x4DBF
        or 0x4E00 <= cp <= 0x9FFF
        or 0xF900 <= cp <= 0xFAFF
    )


def tokenize(text: str) -> list[str]:
    """Lowercase, strip accents (so "ação" matches "acao") and split into word tokens.

    A token is a run of letters, digits, underscores and the combining marks that follow
    them. (Python's `\\w` does not match combining marks, which would split words such as
    Hindi "किताब" at every vowel sign.)
    """
    tokens: list[str] = []
    current: list[str] = []

    def flush() -> None:
        if current:
            tokens.append(unicodedata.normalize("NFC", "".join(current)))
            current.clear()

    for ch in unicodedata.normalize("NFKD", text.casefold()):
        if unicodedata.category(ch).startswith("M"):
            if current and ord(current[-1]) >= _FOLDABLE_BELOW:
                current.append(ch)
            continue
        if _is_cjk(ch):
            flush()
            tokens.append(ch)
        elif ch.isalnum() or ch == "_":
            current.append(ch)
        else:
            flush()
    flush()
    return tokens


def make_tokenizer(stopwords: Iterable[str] = ()) -> Tokenizer:
    """Return a tokenizer that behaves like `tokenize` but drops the given stopwords.

    Stopwords are normalized with the same rules, so "não" also removes "nao".
    """
    stop = {token for word in stopwords for token in tokenize(word)}
    if not stop:
        return tokenize

    def _tokenize(text: str) -> list[str]:
        return [token for token in tokenize(text) if token not in stop]

    return _tokenize
