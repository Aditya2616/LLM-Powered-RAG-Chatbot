from rag_chatbot.text import (
    contains_phrase,
    is_abstention,
    normalise,
    snippet,
    stem,
    token_support,
    tokens,
)


def test_normalise_currency_percent_and_thousands():
    assert (
        normalise("£4.95 and 2,500 pounds, 10% off")
        == "4.95 pounds and 2500 pounds, 10 percent off"
    )
    assert normalise("30-day window") == "30 day window"


def test_contains_phrase_is_whole_word():
    assert contains_phrase("within 30 days of delivery", "30 days")
    assert not contains_phrase("within 130 days", "30 days")
    assert contains_phrase("It costs £12 to collect", "12 pounds")
    assert not contains_phrase("lift 20 kg", "2")
    assert contains_phrase("limited to 2 returns", "2")


def test_is_abstention_variants():
    assert is_abstention("I don't know")
    assert is_abstention("I don't know.")
    assert is_abstention("  i do not know based on the context")
    assert is_abstention("I don’t know")
    assert not is_abstention("I know the answer: 30 days")
    assert not is_abstention("30 days")


def test_stem_only_plurals():
    assert stem("refunds") == "refund"
    assert stem("class") == "class"
    assert stem("this") == "this"
    assert stem("days") == "day"


def test_tokens_keep_numbers_whole():
    assert "4.95" in tokens("costs 4.95 pounds")
    assert "10:15" in tokens("at 10:15")


def test_token_support():
    ctx = "Refunds go to the original payment method within 5 working days."
    assert token_support("5 working days", ctx) == 1.0
    assert token_support("7 working days", ctx) < 1.0
    assert token_support("", ctx) == 1.0


def test_snippet_cuts_on_a_word():
    s = snippet("alpha beta gamma delta", 12)
    assert s.endswith("...") and "gam" not in s
