from model.tokenizer import CharTokenizer


def test_encode_decode_roundtrip():
    tok = CharTokenizer.from_text("hello world")
    ids = tok.encode("hello world")
    assert tok.decode(ids) == "hello world"
    assert all(isinstance(i, int) for i in ids)


def test_vocab_is_sorted_unique_characters():
    tok = CharTokenizer.from_text("banana")
    assert tok.chars == ["a", "b", "n"]
    assert tok.vocab_size == 3


def test_unknown_characters_are_skipped_and_reported():
    tok = CharTokenizer.from_text("abc")
    assert tok.encode("abz") == [0, 1]
    assert tok.unknown_chars("abz") == {"z"}


def test_serialisation_roundtrip():
    tok = CharTokenizer.from_text("MiniLM rocks")
    again = CharTokenizer.from_dict(tok.to_dict())
    assert again.chars == tok.chars
    assert again.encode("rocks") == tok.encode("rocks")
