"""Unicode-aware tokenizer: case folding, accent stripping and CJK splitting."""

import re
import unicodedata
from collections.abc import Callable, Iterable

Tokenizer = Callable[[str], list[str]]

# Chinese and Japanese are written without spaces, so each ideograph or kana is a token.
_CJK = "぀-ヿ㐀-䶿一-鿿豈-﫿"
_TOKEN = re.compile(rf"[{_CJK}]|[^\W{_CJK}]+")
# Accents are only stripped from Latin, Greek and Cyrillic letters; in scripts such as
# Devanagari or Thai the combining marks are vowels and change the word.
_FOLDABLE_BELOW = 0x0530


def tokenize(text: str) -> list[str]:
    """Lowercase, strip accents (so "ação" matches "acao") and split into word tokens."""
    kept: list[str] = []
    for ch in unicodedata.normalize("NFKD", text.casefold()):
        if unicodedata.combining(ch) and kept and ord(kept[-1]) < _FOLDABLE_BELOW:
            continue
        kept.append(ch)
    return _TOKEN.findall(unicodedata.normalize("NFC", "".join(kept)))


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
