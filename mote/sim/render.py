"""Locale renderers (en/ru/ja) for traces: narrative docs, QA pairs, DPO pairs.

Entity keys are locale-independent; these tables map them to words. Russian carries the
accusative/prepositional forms and verb gender agreement it needs; Japanese is particle-regular.
Adding a locale = adding tables + a LOCALES entry (the sim never changes).
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from .domains import Q, Trace

GENDER = {  # for Russian verb agreement
    "mara": "f", "jon": "m", "ivy": "f", "tomas": "m", "lena": "f", "kofi": "m", "sana": "f", "rui": "m",
    "ada": "f", "bruno": "m", "cora": "f", "dario": "m", "elsa": "f", "felix": "m", "greta": "f",
    "hugo": "m", "irene": "f", "janos": "m", "katia": "f", "luca": "m",
}

EN_GOODS_ONE = {"apples": "apple", "loaves": "loaf", "candles": "candle", "nails": "nail", "eggs": "egg"}


def en_n(n: int, g: str) -> str:
    return f"{n} {EN_GOODS_ONE[g] if n == 1 else EN['goods'][g]}"


EN = {
    "rooms": {"kitchen": "the kitchen", "garden": "the garden", "study": "the study", "hall": "the hall",
              "cellar": "the cellar", "attic": "the attic"},
    "objects": {"key": "the key", "book": "the book", "lamp": "the lamp", "coin": "the coin",
                "apple": "the apple", "letter": "the letter", "cup": "the cup", "knife": "the knife"},
    "containers": {"box": "the box", "basket": "the basket", "drawer": "the drawer", "chest": "the chest"},
    "goods": {"apples": "apples", "loaves": "loaves", "candles": "candles", "nails": "nails", "eggs": "eggs"},
    "titles": {"standup": "the standup", "review": "the review", "lunch": "lunch",
               "planning": "the planning meeting", "call": "the call", "workshop": "the workshop"},
}

RU_ROOMS = {"kitchen": ("кухня", "кухню", "кухне"), "garden": ("сад", "сад", "саду"),
            "study": ("кабинет", "кабинет", "кабинете"), "hall": ("зал", "зал", "зале"),
            "cellar": ("подвал", "подвал", "подвале"), "attic": ("чердак", "чердак", "чердаке")}
RU_OBJ = {"key": ("ключ", "ключ"), "book": ("книга", "книгу"), "lamp": ("лампа", "лампу"),
          "coin": ("монета", "монету"), "apple": ("яблоко", "яблоко"), "letter": ("письмо", "письмо"),
          "cup": ("чашка", "чашку"), "knife": ("нож", "нож")}
RU_CONT = {"box": ("коробка", "коробку", "коробке"), "basket": ("корзина", "корзину", "корзине"),
           "drawer": ("ящик", "ящик", "ящике"), "chest": ("сундук", "сундук", "сундуке")}
RU_GOODS = {"apples": "яблок", "loaves": "буханок", "candles": "свечей", "nails": "гвоздей", "eggs": "яиц"}
# numeral agreement: 1 -> singular, 2-4 -> paucal, 5+ -> genitive plural
# (nominative singular, paucal, genitive plural, accusative singular — feminine nouns differ after verbs)
RU_GOODS_FORMS = {"apples": ("яблоко", "яблока", "яблок", "яблоко"), "loaves": ("буханка", "буханки", "буханок", "буханку"),
                  "candles": ("свеча", "свечи", "свечей", "свечу"), "nails": ("гвоздь", "гвоздя", "гвоздей", "гвоздь"),
                  "eggs": ("яйцо", "яйца", "яиц", "яйцо")}
RU_COIN_FORMS = ("монета", "монеты", "монет", "монету")


def ru_n(n: int, forms, acc: bool = False) -> str:
    if n % 10 == 1 and n % 100 != 11:
        return f"{n} {forms[3] if acc else forms[0]}"
    if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        return f"{n} {forms[1]}"
    return f"{n} {forms[2]}"
RU_OBJ_GENDER = {"key": "m", "book": "f", "lamp": "f", "coin": "f", "apple": "n", "letter": "n", "cup": "f", "knife": "m"}
RU_WAS = {"m": "был", "f": "была", "n": "было"}
RU_TITLES = {"standup": "летучка", "review": "ревью", "lunch": "обед", "planning": "планирование",
             "call": "созвон", "workshop": "воркшоп"}

JA = {
    "rooms": {"kitchen": "台所", "garden": "庭", "study": "書斎", "hall": "広間", "cellar": "地下室", "attic": "屋根裏"},
    "objects": {"key": "鍵", "book": "本", "lamp": "ランプ", "coin": "コイン", "apple": "りんご",
                "letter": "手紙", "cup": "カップ", "knife": "ナイフ"},
    "containers": {"box": "箱", "basket": "かご", "drawer": "引き出し", "chest": "チェスト"},
    "goods": {"apples": "りんご", "loaves": "パン", "candles": "ろうそく", "nails": "釘", "eggs": "卵"},
    "titles": {"standup": "朝会", "review": "レビュー", "lunch": "昼食", "planning": "計画会議",
               "call": "電話会議", "workshop": "ワークショップ"},
}
JA_COUNTER = {"apples": "個", "loaves": "個", "candles": "本", "nails": "本", "eggs": "個"}


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:]


def _ru_v(verb_mf: Tuple[str, str], who: str) -> str:
    return verb_mf[1] if GENDER.get(who) == "f" else verb_mf[0]


# ---------------------------------------------------------------- event sentences
def _ev_en(e, kin_names=None) -> str:
    d = e.data
    k = e.kind
    if k == "init_household":
        out = [f"{_cap(p)} is in {EN['rooms'][r]}." for p, r in d["people"].items()]
        out += [f"{_cap(EN['containers'][c])} is in {EN['rooms'][r]}." for c, r in d["containers"].items()]
        out += [f"{_cap(EN['objects'][o])} is in {EN['rooms'][r]}." for o, r in d["objects"].items()]
        return " ".join(out)
    if k == "init_stock":
        out = []
        for p, st in d["start"].items():
            goods = ", ".join(en_n(n, g) for g, n in st["goods"].items())
            out.append(f"{_cap(p)} starts with {goods} and {st['coins']} coins.")
        return " ".join(out)
    if k == "move":
        return f"{_cap(d['who'])} went to {EN['rooms'][d['to']]}."
    if k == "take":
        return f"{_cap(d['who'])} picked up {EN['objects'][d['obj']]}."
    if k == "put_in":
        return f"{_cap(d['who'])} put {EN['objects'][d['obj']]} into {EN['containers'][d['cont']]}."
    if k == "put_down":
        return f"{_cap(d['who'])} put {EN['objects'][d['obj']]} down in {EN['rooms'][d['room']]}."
    if k == "trade":
        coins = "1 coin" if d['cost'] == 1 else f"{d['cost']} coins"
        return f"{_cap(d['buyer'])} bought {en_n(d['n'], d['goods'])} from {_cap(d['seller'])} for {coins}."
    if k == "harvest":
        return f"{_cap(d['who'])} gathered {en_n(d['n'], d['goods'])} more."
    if k == "booked":
        return f"{_cap(d['who'])} booked {EN['titles'][d['title']]} from {d['start_h']}:00 to {d['end_h']}:00."
    if k == "moved":
        return f"{_cap(d['who'])} moved {EN['titles'][d['title']]} from {d['from_h']}:00 to {d['to_h']}:00."
    if k == "family":
        out = [f"{_cap(a)} and {_cap(b)} are married." for a, b in
               {tuple(sorted((x, y))) for x, y in d["spouses"].items() if y}]
        out += [f"{_cap(p1)} and {_cap(p2)} are the parents of {_cap(c)}."
                for c, (p1, p2) in sorted(d["parents"].items())]
        return " ".join(out)
    return ""


def _ev_ru(e) -> str:
    d = e.data
    k = e.kind
    if k == "init_household":
        out = [f"{_cap(p)} в {RU_ROOMS[r][2]}." for p, r in d["people"].items()]
        out += [f"{_cap(RU_CONT[c][0])} в {RU_ROOMS[r][2]}." for c, r in d["containers"].items()]
        out += [f"{_cap(RU_OBJ[o][0])} в {RU_ROOMS[r][2]}." for o, r in d["objects"].items()]
        return " ".join(out)
    if k == "init_stock":
        out = []
        for p, st in d["start"].items():
            goods = ", ".join(ru_n(n, RU_GOODS_FORMS[g]) for g, n in st["goods"].items())
            out.append(f"У {_cap(p)} вначале {goods} и {ru_n(st['coins'], RU_COIN_FORMS)}.")
        return " ".join(out)
    if k == "move":
        return f"{_cap(d['who'])} {_ru_v(('пошёл', 'пошла'), d['who'])} в {RU_ROOMS[d['to']][1]}."
    if k == "take":
        return f"{_cap(d['who'])} {_ru_v(('взял', 'взяла'), d['who'])} {RU_OBJ[d['obj']][1]}."
    if k == "put_in":
        return f"{_cap(d['who'])} {_ru_v(('положил', 'положила'), d['who'])} {RU_OBJ[d['obj']][1]} в {RU_CONT[d['cont']][1]}."
    if k == "put_down":
        return f"{_cap(d['who'])} {_ru_v(('оставил', 'оставила'), d['who'])} {RU_OBJ[d['obj']][1]} в {RU_ROOMS[d['room']][2]}."
    if k == "trade":
        return (f"{_cap(d['buyer'])} {_ru_v(('купил', 'купила'), d['buyer'])} {ru_n(d['n'], RU_GOODS_FORMS[d['goods']], acc=True)} "
                f"у {_cap(d['seller'])} за {ru_n(d['cost'], RU_COIN_FORMS, acc=True)}.")
    if k == "harvest":
        return f"{_cap(d['who'])} {_ru_v(('собрал', 'собрала'), d['who'])} ещё {ru_n(d['n'], RU_GOODS_FORMS[d['goods']], acc=True)}."
    if k == "booked":
        return f"{_cap(d['who'])} {_ru_v(('назначил', 'назначила'), d['who'])} встречу «{RU_TITLES[d['title']]}» с {d['start_h']}:00 до {d['end_h']}:00."
    if k == "moved":
        return f"{_cap(d['who'])} {_ru_v(('перенёс', 'перенесла'), d['who'])} встречу «{RU_TITLES[d['title']]}» с {d['from_h']}:00 на {d['to_h']}:00."
    if k == "family":
        out = [f"{_cap(a)} и {_cap(b)} женаты." for a, b in
               {tuple(sorted((x, y))) for x, y in d["spouses"].items() if y}]
        out += [f"{_cap(p1)} и {_cap(p2)} — родители {_cap(c)}." for c, (p1, p2) in sorted(d["parents"].items())]
        return " ".join(out)
    return ""


def _ev_ja(e) -> str:
    d = e.data
    k = e.kind
    if k == "init_household":
        out = [f"{_cap(p)}は{JA['rooms'][r]}にいる。" for p, r in d["people"].items()]
        out += [f"{JA['containers'][c]}は{JA['rooms'][r]}にある。" for c, r in d["containers"].items()]
        out += [f"{JA['objects'][o]}は{JA['rooms'][r]}にある。" for o, r in d["objects"].items()]
        return "".join(out)
    if k == "init_stock":
        out = []
        for p, st in d["start"].items():
            goods = "、".join(f"{JA['goods'][g]}{n}{JA_COUNTER[g]}" for g, n in st["goods"].items())
            out.append(f"{_cap(p)}は最初、{goods}とコイン{st['coins']}枚を持っている。")
        return "".join(out)
    if k == "move":
        return f"{_cap(d['who'])}は{JA['rooms'][d['to']]}へ移動した。"
    if k == "take":
        return f"{_cap(d['who'])}は{JA['objects'][d['obj']]}を手に取った。"
    if k == "put_in":
        return f"{_cap(d['who'])}は{JA['objects'][d['obj']]}を{JA['containers'][d['cont']]}に入れた。"
    if k == "put_down":
        return f"{_cap(d['who'])}は{JA['objects'][d['obj']]}を{JA['rooms'][d['room']]}に置いた。"
    if k == "trade":
        c = JA_COUNTER[d['goods']]
        return f"{_cap(d['buyer'])}は{_cap(d['seller'])}から{JA['goods'][d['goods']]}を{d['n']}{c}、{d['cost']}枚のコインで買った。"
    if k == "harvest":
        return f"{_cap(d['who'])}は{JA['goods'][d['goods']]}をさらに{d['n']}{JA_COUNTER[d['goods']]}集めた。"
    if k == "booked":
        return f"{_cap(d['who'])}は{d['start_h']}時から{d['end_h']}時まで{JA['titles'][d['title']]}を予定に入れた。"
    if k == "moved":
        return f"{_cap(d['who'])}は{JA['titles'][d['title']]}を{d['from_h']}時から{d['to_h']}時に変更した。"
    if k == "family":
        out = [f"{_cap(a)}と{_cap(b)}は夫婦だ。" for a, b in
               {tuple(sorted((x, y))) for x, y in d["spouses"].items() if y}]
        out += [f"{_cap(p1)}と{_cap(p2)}は{_cap(c)}の両親だ。" for c, (p1, p2) in sorted(d["parents"].items())]
        return "".join(out)
    return ""


# ---------------------------------------------------------------- questions + answers
def _place_en(ans) -> str:
    tag, v = ans
    if tag == "room":
        return f"in {EN['rooms'][v]}"
    if tag == "cont":
        return f"in {EN['containers'][v]}"
    return f"with {_cap(v)}"  # held


def _qa_en(q: Q) -> Tuple[str, str, str]:
    a = q.args
    if q.qtype == "where_obj":
        return (f"Where is {EN['objects'][a['obj']]} now?", f"It is {_place_en(q.answer)}.", f"It is {_place_en(q.wrong)}.")
    if q.qtype == "where_obj_start":
        return (f"Where was {EN['objects'][a['obj']]} at the beginning?",
                f"It was {_place_en(q.answer)}.", f"It was {_place_en(q.wrong)}.")
    if q.qtype == "where_person":
        return (f"Where is {_cap(a['who'])} now?", f"{_cap(a['who'])} is in {EN['rooms'][q.answer[1]]}.",
                f"{_cap(a['who'])} is in {EN['rooms'][q.wrong[1]]}.")
    if q.qtype == "count_loose_in_room":
        def there(n):
            return f"There is {n}." if n == 1 else f"There are {n}."
        return (f"How many objects are lying in {EN['rooms'][a['room']]} (not held, not in a container)?",
                there(q.answer[1]), there(q.wrong[1]))
    if q.qtype == "count_goods":
        return (f"How many {EN['goods'][a['goods']]} does {_cap(a['who'])} have now?",
                f"{_cap(a['who'])} has {q.answer[1]}.", f"{_cap(a['who'])} has {q.wrong[1]}.")
    if q.qtype == "count_coins":
        return (f"How many coins does {_cap(a['who'])} have now?",
                f"{_cap(a['who'])} has {q.answer[1]} coins.", f"{_cap(a['who'])} has {q.wrong[1]} coins.")
    if q.qtype == "who_has_more":
        return (f"Who has more {EN['goods'][a['goods']]}, {_cap(a['a'])} or {_cap(a['b'])}?",
                f"{_cap(q.answer[1])} does.", f"{_cap(q.wrong[1])} does.")
    if q.qtype == "grandparent":
        return (f"Who is {_cap(a['who'])}'s grandparent through {_cap(a['via'])}?",
                f"{_cap(q.answer[1])}.", f"{_cap(q.wrong[1])}.")
    if q.qtype == "count_siblings":
        return (f"How many siblings does {_cap(a['who'])} have?", f"{q.answer[1]}.", f"{q.wrong[1]}.")
    if q.qtype == "count_children":
        return (f"How many children does {_cap(a['who'])} have?", f"{q.answer[1]}.", f"{q.wrong[1]}.")
    if q.qtype == "spouse_of":
        return (f"Who is {_cap(a['who'])} married to?", f"{_cap(q.answer[1])}.", f"{_cap(q.wrong[1])}.")
    if q.qtype == "free_at":
        yes, no = "Yes.", "No."
        return (f"Is {_cap(a['who'])} free at {a['hour']}:00?", yes if q.answer[1] else no, yes if q.wrong[1] else no)
    if q.qtype == "first_meeting":
        return (f"What is {_cap(a['who'])}'s first meeting of the day?",
                f"{_cap(EN['titles'][q.answer[1]])}.", f"{_cap(EN['titles'][q.wrong[1]])}.")
    if q.qtype == "count_meetings":
        return (f"How many meetings does {_cap(a['who'])} have?", f"{q.answer[1]}.", f"{q.wrong[1]}.")
    if q.qtype == "overlap":
        yes, no = "Yes.", "No."
        return (f"Do {_cap(a['a'])} and {_cap(a['b'])} have overlapping meetings?",
                yes if q.answer[1] else no, yes if q.wrong[1] else no)
    raise KeyError(q.qtype)


def _place_ru(ans) -> str:
    tag, v = ans
    if tag == "room":
        return f"в {RU_ROOMS[v][2]}"
    if tag == "cont":
        return f"в {RU_CONT[v][2]}"
    return f"у {_cap(v)}"


def _qa_ru(q: Q) -> Tuple[str, str, str]:
    a = q.args
    m = {
        "where_obj": (f"Где сейчас {RU_OBJ[a.get('obj', 'key')][0]}?", f"{_cap(RU_OBJ[a.get('obj','key')][0])} {_place_ru(q.answer)}.",
                      f"{_cap(RU_OBJ[a.get('obj','key')][0])} {_place_ru(q.wrong)}.") if q.qtype == "where_obj" else None,
    }
    if q.qtype == "where_obj":
        return m["where_obj"]
    if q.qtype == "where_obj_start":
        o = RU_OBJ[a["obj"]][0]
        was = RU_WAS[RU_OBJ_GENDER[a["obj"]]]
        return (f"Где {o} {was} в начале?", f"{_cap(o)} {was} {_place_ru(q.answer)}.", f"{_cap(o)} {was} {_place_ru(q.wrong)}.")
    if q.qtype == "where_person":
        return (f"Где сейчас {_cap(a['who'])}?", f"{_cap(a['who'])} в {RU_ROOMS[q.answer[1]][2]}.",
                f"{_cap(a['who'])} в {RU_ROOMS[q.wrong[1]][2]}.")
    if q.qtype == "count_loose_in_room":
        return (f"Сколько предметов лежит в {RU_ROOMS[a['room']][2]} (не в руках и не в ёмкости)?",
                f"Там {q.answer[1]}.", f"Там {q.wrong[1]}.")
    if q.qtype == "count_goods":
        return (f"Сколько {RU_GOODS[a['goods']]} сейчас у {_cap(a['who'])}?",
                f"У {_cap(a['who'])} {ru_n(q.answer[1], RU_GOODS_FORMS[a['goods']])}.",
                f"У {_cap(a['who'])} {ru_n(q.wrong[1], RU_GOODS_FORMS[a['goods']])}.")
    if q.qtype == "count_coins":
        return (f"Сколько монет сейчас у {_cap(a['who'])}?",
                f"У {_cap(a['who'])} {ru_n(q.answer[1], RU_COIN_FORMS)}.",
                f"У {_cap(a['who'])} {ru_n(q.wrong[1], RU_COIN_FORMS)}.")
    if q.qtype == "who_has_more":
        return (f"У кого больше {RU_GOODS[a['goods']]} — у {_cap(a['a'])} или у {_cap(a['b'])}?",
                f"У {_cap(q.answer[1])}.", f"У {_cap(q.wrong[1])}.")
    if q.qtype == "grandparent":
        return (f"Кто дедушка или бабушка {_cap(a['who'])} по линии {_cap(a['via'])}?",
                f"{_cap(q.answer[1])}.", f"{_cap(q.wrong[1])}.")
    if q.qtype == "count_siblings":
        return (f"Сколько братьев и сестёр у {_cap(a['who'])}?", f"{q.answer[1]}.", f"{q.wrong[1]}.")
    if q.qtype == "count_children":
        return (f"Сколько детей у {_cap(a['who'])}?", f"{q.answer[1]}.", f"{q.wrong[1]}.")
    if q.qtype == "spouse_of":
        return (f"С кем в браке {_cap(a['who'])}?", f"С {_cap(q.answer[1])}.", f"С {_cap(q.wrong[1])}.")
    if q.qtype == "free_at":
        yes, no = "Да.", "Нет."
        return (f"Свободен(на) ли {_cap(a['who'])} в {a['hour']}:00?", yes if q.answer[1] else no, yes if q.wrong[1] else no)
    if q.qtype == "first_meeting":
        return (f"Какая первая встреча у {_cap(a['who'])}?", f"«{_cap(RU_TITLES[q.answer[1]])}».",
                f"«{_cap(RU_TITLES[q.wrong[1]])}».")
    if q.qtype == "count_meetings":
        return (f"Сколько встреч у {_cap(a['who'])}?", f"{q.answer[1]}.", f"{q.wrong[1]}.")
    if q.qtype == "overlap":
        yes, no = "Да.", "Нет."
        return (f"Пересекаются ли встречи {_cap(a['a'])} и {_cap(a['b'])}?",
                yes if q.answer[1] else no, yes if q.wrong[1] else no)
    raise KeyError(q.qtype)


def _place_ja(ans) -> str:
    tag, v = ans
    if tag == "room":
        return f"{JA['rooms'][v]}にある"
    if tag == "cont":
        return f"{JA['containers'][v]}の中にある"
    return f"{_cap(v)}が持っている"


def _qa_ja(q: Q) -> Tuple[str, str, str]:
    a = q.args
    if q.qtype == "where_obj":
        o = JA["objects"][a["obj"]]
        return (f"{o}は今どこにある？", f"{o}は{_place_ja(q.answer)}。", f"{o}は{_place_ja(q.wrong)}。")
    if q.qtype == "where_obj_start":
        o = JA["objects"][a["obj"]]
        def was(ans):
            if ans[0] == "held":
                return f"最初、{o}は{_cap(ans[1])}が持っていた。"
            return f"最初、{o}は{_place_ja(ans)[:-2]}あった。"
        return (f"{o}は最初どこにあった？", was(q.answer), was(q.wrong))
    if q.qtype == "where_person":
        return (f"{_cap(a['who'])}は今どこにいる？", f"{_cap(a['who'])}は{JA['rooms'][q.answer[1]]}にいる。",
                f"{_cap(a['who'])}は{JA['rooms'][q.wrong[1]]}にいる。")
    if q.qtype == "count_loose_in_room":
        return (f"{JA['rooms'][a['room']]}に置いてある物はいくつ？（手に持たれておらず、入れ物にも入っていない物）",
                f"{q.answer[1]}個。", f"{q.wrong[1]}個。")
    if q.qtype == "count_goods":
        c = JA_COUNTER[a['goods']]
        return (f"{_cap(a['who'])}は今{JA['goods'][a['goods']]}をいくつ持っている？",
                f"{q.answer[1]}{c}。", f"{q.wrong[1]}{c}。")
    if q.qtype == "count_coins":
        return (f"{_cap(a['who'])}は今コインを何枚持っている？", f"{q.answer[1]}枚。", f"{q.wrong[1]}枚。")
    if q.qtype == "who_has_more":
        return (f"{JA['goods'][a['goods']]}を多く持っているのは{_cap(a['a'])}と{_cap(a['b'])}のどちら？",
                f"{_cap(q.answer[1])}。", f"{_cap(q.wrong[1])}。")
    if q.qtype == "grandparent":
        return (f"{_cap(a['via'])}を通じた{_cap(a['who'])}の祖父母は誰？", f"{_cap(q.answer[1])}。", f"{_cap(q.wrong[1])}。")
    if q.qtype == "count_siblings":
        return (f"{_cap(a['who'])}の兄弟姉妹は何人？", f"{q.answer[1]}人。", f"{q.wrong[1]}人。")
    if q.qtype == "count_children":
        return (f"{_cap(a['who'])}の子供は何人？", f"{q.answer[1]}人。", f"{q.wrong[1]}人。")
    if q.qtype == "spouse_of":
        return (f"{_cap(a['who'])}の配偶者は誰？", f"{_cap(q.answer[1])}。", f"{_cap(q.wrong[1])}。")
    if q.qtype == "free_at":
        yes, no = "はい、空いている。", "いいえ、空いていない。"
        return (f"{_cap(a['who'])}は{a['hour']}時に空いている？", yes if q.answer[1] else no, yes if q.wrong[1] else no)
    if q.qtype == "first_meeting":
        return (f"{_cap(a['who'])}の最初の予定は何？", f"{JA['titles'][q.answer[1]]}。", f"{JA['titles'][q.wrong[1]]}。")
    if q.qtype == "count_meetings":
        return (f"{_cap(a['who'])}の予定はいくつ？", f"{q.answer[1]}件。", f"{q.wrong[1]}件。")
    if q.qtype == "overlap":
        yes, no = "はい、重なっている。", "いいえ、重なっていない。"
        return (f"{_cap(a['a'])}と{_cap(a['b'])}の予定は重なっている？", yes if q.answer[1] else no, yes if q.wrong[1] else no)
    raise KeyError(q.qtype)


LOCALES = {
    "en": {"event": _ev_en, "qa": _qa_en, "sep": " "},
    "ru": {"event": _ev_ru, "qa": _qa_ru, "sep": " "},
    "ja": {"event": _ev_ja, "qa": _qa_ja, "sep": ""},
}


def narrative(trace: Trace, locale: str) -> str:
    L = LOCALES[locale]
    return L["sep"].join(s for s in (L["event"](e) for e in trace.events) if s)


def qa_pairs(trace: Trace, locale: str) -> List[Dict[str, Any]]:
    L = LOCALES[locale]
    out = []
    for q in trace.questions:
        qt, ans, wrong = L["qa"](q)
        out.append({"question": qt, "answer": ans, "wrong": wrong,
                    "qtype": q.qtype, "wrong_kind": q.wrong_kind})
    return out
