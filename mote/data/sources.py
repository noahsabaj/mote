"""Dataset sources for the flagship mixes (researched 2026-08-22; all ungated on Hugging Face).

Each pretraining source yields plain-text documents; each SFT source yields message lists.
Percentages are the agreed mix; the builders scale them to a byte budget.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable, Dict, Iterator, List, Optional


@dataclass
class PretrainSource:
    key: str
    path: str
    share: float  # fraction of the pretraining byte budget
    name: Optional[str] = None
    split: str = "train"
    text: Callable[[dict], Optional[str]] = lambda r: r.get("text")
    keep: Callable[[dict], bool] = lambda r: True
    note: str = ""
    min_bytes: Optional[int] = None  # override the builder's window (long-document sources)
    max_bytes: Optional[int] = None
    chunk: bool = False  # split over-long documents at paragraph breaks instead of dropping them (books)


def _fineweb_keep(r: dict) -> bool:
    return r.get("int_score", 0) >= 3


def _synth_text(r: dict) -> Optional[str]:
    q, a = r.get("query"), r.get("synthetic_answer")
    if not q or not a:
        return None
    return f"Question: {q.strip()}\n\nAnswer: {a.strip()}"


def _finephrase_text(r: dict) -> Optional[str]:
    # keep only the rephrased completion; the source text is FineWeb-Edu and would duplicate it
    rr = r.get("rollout_results") or r.get("rollout") or r.get("completion")
    if isinstance(rr, list) and rr:
        rr = rr[0]
    if isinstance(rr, dict):
        rr = rr.get("text") or rr.get("completion") or rr.get("content")
    return rr if isinstance(rr, str) else None


def _cosmopedia_keep(r: dict) -> bool:
    fmt = (r.get("format") or "").lower()
    aud = (r.get("audience") or "").lower()
    return any(k in fmt for k in ("story", "wikihow", "textbook")) and any(k in aud for k in ("young", "general", "grade"))


PRETRAIN: List[PretrainSource] = [
    PretrainSource("fineweb_edu", "HuggingFaceFW/fineweb-edu", 0.35, name="sample-10BT", keep=_fineweb_keep, note="edu web, int_score>=3"),
    PretrainSource("dclm_edu", "HuggingFaceTB/dclm-edu", 0.15, note="DCLM filtered with the FineWeb-Edu classifier"),
    PretrainSource("finephrase", "HuggingFaceFW/finephrase", 0.15, name="faq", text=_finephrase_text, note="rephrased completions only"),
    PretrainSource("ultra_fineweb_l3", "openbmb/Ultra-FineWeb-L3", 0.10, name="Ultra-FineWeb-L3-en-Multi-Style-Synthetic", text=lambda r: r.get("content"), note="short rewrites, 100% <= 4 KB"),
    PretrainSource("synth", "PleIAs/SYNTH", 0.10, text=_synth_text, keep=lambda r: (r.get("language") or "en").startswith("en"), note="query+answer rendered as Q/A"),
    PretrainSource("cosmopedia_v2", "HuggingFaceTB/smollm-corpus", 0.08, name="cosmopedia-v2", keep=_cosmopedia_keep, note="stories/wikihow/textbook"),
    PretrainSource("nemotron_factseek", "nvidia/Nemotron-Pretraining-Specialized-v1.2", 0.04, name="Nemotron-Pretraining-Fact-Seeking", note="tiny QA docs"),
    PretrainSource("finewiki_simple", "HuggingFaceFW/finewiki", 0.03, name="simple", note="CC-BY-SA"),
]


def _permissive_code(r: dict) -> bool:
    lic = str(r.get("license") or "").lower()
    return any(k in lic for k in ("mit", "apache", "bsd", "isc", "unlicense"))


def _fw2(lang: str, share: float) -> "PretrainSource":
    return PretrainSource(f"fw2_{lang[:3]}", "HuggingFaceFW/fineweb-2", share, name=lang,
                          note=f"multilingual slice {lang}")


# The flagship recipe (grilled 2026-08-23, docs/shape.md): backbone ~63%, reasoning 11%, code 8%,
# multilingual 9% (es/fr/de/pt/ru/ja), long documents 10%. Long sources carry their own byte windows;
# the builder also writes per-domain val shards for the trade-offs to stay visible.
FLAGSHIP: List[PretrainSource] = [
    PretrainSource("fineweb_edu", "HuggingFaceFW/fineweb-edu", 0.25, name="sample-10BT", keep=_fineweb_keep, note="edu web, int_score>=3"),
    PretrainSource("dclm_edu", "HuggingFaceTB/dclm-edu", 0.09, note="DCLM filtered with the FineWeb-Edu classifier"),
    PretrainSource("finephrase", "HuggingFaceFW/finephrase", 0.07, name="faq", text=_finephrase_text, note="rephrased completions only"),
    PretrainSource("ultra_fineweb_l3", "openbmb/Ultra-FineWeb-L3", 0.05, name="Ultra-FineWeb-L3-en-Multi-Style-Synthetic", text=lambda r: r.get("content"), note="short rewrites"),
    PretrainSource("synth", "PleIAs/SYNTH", 0.08, text=_synth_text, keep=lambda r: (r.get("language") or "en").startswith("en"), note="query+answer as Q/A"),
    PretrainSource("cosmopedia_v2", "HuggingFaceTB/smollm-corpus", 0.07, name="cosmopedia-v2", keep=_cosmopedia_keep, note="stories/wikihow/textbook"),
    PretrainSource("nemotron_factseek", "nvidia/Nemotron-Pretraining-Specialized-v1.2", 0.02, name="Nemotron-Pretraining-Fact-Seeking", note="tiny QA docs"),
    PretrainSource("finewiki_simple", "HuggingFaceFW/finewiki", 0.01, name="simple", note="CC-BY-SA"),
    # reasoning
    PretrainSource("finemath", "HuggingFaceTB/finemath", 0.10, name="finemath-3plus", note="math/step-by-step-dense web"),
    # code (codeparrot-clean: inline content + license; github-code-clean and stack-edu ship scripts/blob-ids only)
    PretrainSource("code", "codeparrot/codeparrot-clean", 0.07, text=lambda r: r.get("content"), keep=_permissive_code, note="Python, permissive licenses"),
    PretrainSource("code_long", "codeparrot/codeparrot-clean", 0.01, text=lambda r: r.get("content"), keep=_permissive_code, min_bytes=8192, max_bytes=65536, chunk=True, note="long files, chunked"),
    # multilingual (two extra scripts on purpose: byte-level UTF-8 is home turf)
    _fw2("spa_Latn", 0.015), _fw2("fra_Latn", 0.015), _fw2("deu_Latn", 0.015),
    _fw2("por_Latn", 0.015), _fw2("rus_Cyrl", 0.015), _fw2("jpn_Jpan", 0.015),
    # long documents (teach the 16384 window something real)
    PretrainSource("finewiki_long", "HuggingFaceFW/finewiki", 0.04, name="en", min_bytes=8192, max_bytes=65536, note="full articles"),
    PretrainSource("gutenberg", "manu/project_gutenberg", 0.03, split="en", min_bytes=8192, max_bytes=65536, chunk=True, note="books, chunked at paragraph breaks"),
    PretrainSource("fineweb_long", "HuggingFaceFW/fineweb-edu", 0.02, name="sample-10BT", keep=_fineweb_keep, min_bytes=8192, max_bytes=65536, note="long edu pages"),
]
assert abs(sum(s.share for s in FLAGSHIP) - 1.0) < 1e-9, sum(s.share for s in FLAGSHIP)

# The anneal (cooldown) composition, signed 2026-08-24 (docs/shape.md § pipeline): the same sources as
# FLAGSHIP with the weight moved toward math, Q&A and rewrites — the "quality annealing" of OLMo 2 /
# Llama 3 / OctoThinker. Fresh documents come from `build_mix --list anneal --skip-after <earlier metas>`.
# The cooldown branch adds plain-LM extras on top via `--mix …:plain` (sim narrative+QA 4 %, chat 3 %,
# identity 0.2 %), so these weights are of the remaining ~93 % of the branch's bytes.
_ANNEAL_WEIGHTS: Dict[str, float] = {
    "fineweb_edu": 18, "dclm_edu": 8, "ultra_fineweb_l3": 4, "nemotron_factseek": 2, "finewiki_simple": 1,
    "synth": 10, "finephrase": 7, "cosmopedia_v2": 7,
    "finemath": 15,
    "code": 7, "code_long": 1,
    "fw2_spa": 1, "fw2_fra": 1, "fw2_deu": 1, "fw2_por": 1, "fw2_rus": 1, "fw2_jpn": 1,
    # Long documents, raised 2026-08-26 from 3/2/2 (docs/research/midtraining-2026-08-26.md). As first
    # written the reweighting cut the long-document share from FLAGSHIP's 10.0 % to 8.6 % — a 14 % relative
    # cut, in the same direction as the failure PRISM (2603.17074 §8.1) measured at larger amplitude, where
    # short-context mid-training took RULER@128k from 59.09 to 6.46. OctoLong (2608.05141) argues the
    # opposite move: replacing part of a context-extension mixture with dependency-dense material improved
    # long-range retrieval and state tracking. So the natural long sources go back above FLAGSHIP's share,
    # and `data/sim_long` supplies the dependency-dense part from the extras side of the mix.
    "finewiki_long": 4, "gutenberg": 3, "fineweb_long": 2,
}
assert set(_ANNEAL_WEIGHTS) == {s.key for s in FLAGSHIP}, set(_ANNEAL_WEIGHTS) ^ {s.key for s in FLAGSHIP}
ANNEAL: List[PretrainSource] = [replace(s, share=_ANNEAL_WEIGHTS[s.key] / sum(_ANNEAL_WEIGHTS.values())) for s in FLAGSHIP]


@dataclass
class SFTSource:
    key: str
    path: str
    share: float
    name: Optional[str] = None
    split: str = "train"
    messages: Callable[[dict], Optional[List[dict]]] = lambda r: r.get("messages")
    keep: Callable[[dict], bool] = lambda r: True
    note: str = ""


def _sharegpt(r: dict) -> Optional[List[dict]]:
    conv = r.get("conversations") or r.get("conversation")
    if not conv:
        return None
    out = []
    for m in conv:
        role = m.get("from") or m.get("role")
        content = m.get("value") or m.get("content")
        if role in ("human", "user"):
            out.append({"role": "user", "content": content})
        elif role in ("gpt", "assistant"):
            out.append({"role": "assistant", "content": content})
        elif role == "system":
            out.append({"role": "system", "content": content})
    return out or None


def _wildchat(r: dict) -> Optional[List[dict]]:
    conv = r.get("conversation")
    if not conv:
        return None
    return [{"role": m["role"], "content": m["content"]} for m in conv if m.get("role") in ("user", "assistant")]


def _oasst_keep(r: dict) -> bool:
    return (r.get("lang") == "en") and (r.get("role") in ("prompter", "assistant"))


# Nemotron-Post-Training-Dataset-v1 'chat' was dropped: its user prompts are redacted (empty) on the Hub
# (the originals live in gated LMSYS-Chat-1M), so every row failed the empty-turn filter.
SFT: List[SFTSource] = [
    SFTSource("smol_smoltalk", "HuggingFaceTB/smol-smoltalk", 0.44, note="SmolLM2 recipe"),
    SFTSource("hermes3", "NousResearch/Hermes-3-Dataset", 0.23, messages=_sharegpt, keep=lambda r: len(r.get("conversations", [])) >= 4, note="multi-turn"),
    SFTSource("smoltalk2_systemchats", "HuggingFaceTB/smoltalk2", 0.07, name="SFT", split="smoltalk_smollm3_systemchats_30k_no_think", note="no_think subset"),
    SFTSource("smoltalk2_everyday", "HuggingFaceTB/smoltalk2", 0.03, name="SFT", split="smoltalk_smollm3_everyday_conversations_no_think", note="no_think subset"),
    SFTSource("wildchat", "allenai/WildChat-4.8M", 0.10, messages=_wildchat, keep=lambda r: r.get("language") == "English" and not r.get("toxic") and not r.get("redacted"), note="real user prompts"),
    SFTSource("dolci", "allenai/Dolci-Instruct-SFT", 0.10, keep=lambda r: (r.get("domain") or "") not in ("tool_use", "code"), note="chat/IF domains"),
    SFTSource("oasst2", "OpenAssistant/oasst2", 0.03, messages=None, keep=_oasst_keep, note="threads assembled by the builder"),
]
