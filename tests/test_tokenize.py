import pytest

from bm25_tiny import make_tokenizer, tokenize


def test_lowercases_and_strips_accents():
    assert tokenize("Ação RÁPIDA, café!") == ["acao", "rapida", "cafe"]


def test_splits_cjk_by_character():
    assert tokenize("東京タワー tokyo") == ["東", "京", "タ", "ワ", "ー", "tokyo"]


def test_keeps_vowel_marks_in_non_latin_scripts():
    # Devanagari vowel signs are combining marks but change the word.
    assert tokenize("किताब") == ["किताब"]


def test_keeps_digits_and_underscores():
    assert tokenize("python3 snake_case 2026") == ["python3", "snake_case", "2026"]


def test_empty_text():
    assert tokenize("  ...  ") == []


def test_stopwords_are_normalized_like_text():
    tok = make_tokenizer(["Não", "the"])
    assert tok("The cat não nao dorme") == ["cat", "dorme"]


def test_no_stopwords_returns_plain_tokenizer():
    assert make_tokenizer() is tokenize


def test_keeps_kana_voicing_marks():
    assert tokenize("がか") == ["が", "か"]
    assert tokenize("ガイドabc") == ["ガ", "イ", "ド", "abc"]


def test_stopwords_reject_single_string():
    with pytest.raises(TypeError):
        make_tokenizer("the")
