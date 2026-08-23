"""Folding (morpheme.serve.context): fit guarantees, fold points, the card's contents, the modes."""

from morpheme.serve.context import build_card, context_report, fold, user_facts
from morpheme.tokenizer import ByteTokenizer, ChatMessage

TOK = ByteTokenizer()
CARD = {"role": "system", "content": "You are Mote, a small byte-level model."}


def convo(n_pairs: int, first="My dog's name is Biscuit. I live in Lisbon. Do you like dogs?"):
    msgs = [CARD, {"role": "user", "content": first}, {"role": "assistant", "content": "Noted."}]
    for i in range(n_pairs):
        msgs.append({"role": "user", "content": f"Tell me something about topic number {i} please."})
        msgs.append({"role": "assistant", "content": f"Topic {i} is a thing people discuss at length in many places, sentence after sentence."})
    msgs.append({"role": "user", "content": "What is my dog's name?"})
    return msgs


def test_short_conversation_is_untouched():
    f = fold(convo(1), 2048, 256, TOK)
    assert f.folded_from is None and f.card is None and not f.truncated
    assert f.used == len(TOK.format_chat([ChatMessage(m["role"], m["content"]) for m in convo(1)]))


def test_fold_fits_and_keeps_first_message_and_facts():
    msgs = convo(30)
    f = fold(msgs, 1024, 128, TOK)
    assert f.used <= 1024 - 128
    assert f.folded_from is not None and msgs[1:][f.folded_from]["role"] == "user"
    assert "Biscuit" in f.card and "I live in Lisbon." in f.card
    assert "Do you like dogs?" not in f.card.split("Notes:")[-1]  # questions are not facts
    # the card rides inside the first kept user turn, and the final question is verbatim
    text = bytes(b for b in f.ids if b < 256).decode("utf-8", errors="replace")
    assert "Earlier in this conversation" in text and text.rstrip().endswith("What is my dog's name?")


def test_off_is_plain_truncation():
    f = fold(convo(30), 1024, 128, TOK, mode="off")
    assert f.truncated and f.card is None and f.folded_from is None and f.used <= 1024 - 128


def test_now_folds_everything_before_the_last_user_turn():
    msgs = convo(2)
    f = fold(msgs, 2048, 256, TOK, mode="now")
    rest = msgs[1:]
    assert f.folded_from == max(k for k, m in enumerate(rest) if m["role"] == "user")
    assert f.card and "Biscuit" in f.card


def test_card_override_is_used_verbatim():
    f = fold(convo(30), 1024, 128, TOK, card_override="(Earlier: the user's dog is Biscuit.)")
    assert f.card == "(Earlier: the user's dog is Biscuit.)" and f.used <= 1024 - 128


def test_giant_single_message_truncates_honestly():
    msgs = [CARD, {"role": "user", "content": "x" * 5000}]
    f = fold(msgs, 1024, 128, TOK)
    assert f.truncated


def test_user_facts_rules():
    ms = [ChatMessage("user", "I think it's fine. My sister is called Mara! Where is Lisbon? I'm not sure. I drive a red Fiat."),
          ChatMessage("assistant", "My name is Mote."), ChatMessage("user", "I drive a red Fiat.")]
    assert user_facts(ms) == ["My sister is called Mara!", "I drive a red Fiat."]
    assert "Mara" in build_card(ms)


def test_report_shape():
    r = context_report(convo(30), 1024, 128, TOK)
    assert set(r) == {"used", "limit", "reserve", "fold", "truncated"} and r["fold"]["from"] == r["fold"]["turns"]
