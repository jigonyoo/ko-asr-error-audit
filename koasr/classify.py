"""M3: label each ref/hyp difference by what kind of error it is.

The alignment runs on text normalized up to (not including) the spacing layer,
so pure punctuation and numeral-format differences are already gone and every
remaining chunk is a real difference in words.

Each differing chunk gets one category. Categories carry a severity:
  format     the difference is about writing only
  ambiguous  a person has to listen or know context to decide
  meaning    the sentence now says something else

These are heuristics. They are validated two ways: against planted defects
(koasr/plant.py) and against human judgments (annotations/judgments.jsonl).
Both results are published, including the misses.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

import jiwer

from .normalize import apply_upto, nospace

SEVERITY = {
    "spacing": "format",
    "latin_transliteration": "ambiguous",
    "particle": "ambiguous",
    "minor_deletion": "ambiguous",
    "minor_insertion": "ambiguous",
    "negation": "meaning",
    "number": "meaning",
    "native_numeral": "ambiguous",
    "sound_alike": "meaning",
    "lexical": "meaning",
    "omission": "meaning",
    "addition": "meaning",
}
RANK = {"format": 0, "ambiguous": 1, "meaning": 2}

NEGATION = re.compile(r"^(안|못)$|않|없|아니|못하|말(라|고|자)")
NUM_CHARS = set("영일이삼사오육칠팔구십백천만억조점")
UNIT_CHARS = set("십백천만억조")
COUNTER = re.compile(r"(퍼센트|센티미터|킬로미터|밀리미터|킬로그램|분기|원|년|월|시|분|초|개|명|위|대|층|번|회)$")
PARTICLES = ("에서", "으로", "에게", "한테", "께서", "까지", "부터", "만이", "은", "는", "이", "가",
             "을", "를", "에", "의", "도", "만", "로", "와", "과", "다", "요")


def _jamo(s: str) -> str:
    return unicodedata.normalize("NFD", s)


def _lev(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _strip_particle(w: str) -> str:
    for p in PARTICLES:
        if w.endswith(p) and len(w) > len(p):
            return w[: -len(p)]
    return w


def _numberish(words: list[str]) -> list[bool]:
    """Which words are (part of) a Sino-Korean number, e.g. 이천 구백 십 구 억원.

    A word counts if, once a trailing counter (원, 년, 퍼센트, ...) is removed, it is
    made only of numeral syllables and contains a unit (십백천만억조). A bare digit
    word (구, 삼) counts only next to such a word, because 이 and 일 are also
    ordinary words. 만이 ('only') is not a number: 이 there is a particle.
    """
    cores = [COUNTER.sub("", w) for w in words]
    strong = [bool(c) and set(c) <= NUM_CHARS and bool(set(c) & UNIT_CHARS) for c in cores]
    out = []
    for i, c in enumerate(cores):
        weak = bool(c) and len(c) == 1 and c in NUM_CHARS and (
            (i > 0 and strong[i - 1]) or (i + 1 < len(cores) and strong[i + 1]))
        out.append(strong[i] or weak)
    return out


def _number_signature(words: list[str]) -> str:
    flags = _numberish(words)
    return "".join(COUNTER.sub("", w) for w, f in zip(words, flags) if f)


NATIVE = {"하나", "한", "둘", "두", "셋", "세", "넷", "네", "다섯", "여섯", "일곱", "여덟", "아홉",
          "열", "스물", "스무", "서른", "마흔", "쉰", "예순", "일흔", "여든", "아흔"}


def _has_native(words: list[str]) -> bool:
    return any(COUNTER.sub("", w) in NATIVE for w in words)


def _native_only(a: list[str], b: list[str]) -> bool:
    """True if the span differs only by a native vs Sino numeral (스물 센티미터를 / 이십센티미터 를)."""
    for x, y in ((a, b), (b, a)):
        if _has_native(x) and not _has_native(y):
            rest_x = "".join(w for w in x if COUNTER.sub("", w) not in NATIVE)
            rest_y = "".join(w.lstrip("영일이삼사오육칠팔구십백천만억조") for w in y)
            if rest_x and rest_x == rest_y:
                return True
    return False


def _neg_count(words: list[str]) -> int:
    return sum(1 for w in words if NEGATION.search(w))


def categorize(ref_words: list[str], hyp_words: list[str]) -> str:
    r, h = " ".join(ref_words), " ".join(hyp_words)
    if nospace(r) == nospace(h):
        return "spacing"
    # Negation first: a lone inserted 안 is a one-word insertion, but it flips meaning.
    if _neg_count(ref_words) != _neg_count(hyp_words):
        return "negation"
    if not hyp_words:
        return "omission" if len(ref_words) >= 3 else "minor_deletion"
    if not ref_words:
        return "addition" if len(hyp_words) >= 3 else "minor_insertion"
    if re.search(r"[A-Za-z]", r + h):
        return "latin_transliteration"
    # 아홉 시 vs 9시 -> 구시: same number, native vs Sino reading. The layer that
    # would fix it is not written (see normalize.py), so flag it, don't punish it.
    for a, b in ((ref_words, hyp_words), (hyp_words, ref_words)):
        if _has_native(a) and not _has_native(b) and any(w[:1] in "일이삼사오육칠팔구십" for w in b):
            return "native_numeral"
    if _number_signature(ref_words) != _number_signature(hyp_words):
        return "number"
    rs, hs = nospace(r), nospace(h)
    if _strip_particle(rs) == _strip_particle(hs) and rs != hs:
        return "particle"
    jr, jh = _jamo(rs), _jamo(hs)
    if _lev(jr, jh) <= max(2, len(jr) // 4):
        return "sound_alike"
    return "lexical"


@dataclass
class Chunk:
    ref: str
    hyp: str
    category: str

    @property
    def severity(self) -> str:
        return SEVERITY[self.category]


def classify(ref: str, hyp: str) -> list[Chunk]:
    r = apply_upto(ref, 4).split()
    h = apply_upto(hyp, 4).split()
    if not r and not h:
        return []
    out = jiwer.process_words(" ".join(r) or " ", " ".join(h) or " ")
    # Group adjacent non-equal alignment ops. A spacing error shows up as a
    # substitution next to a deletion, so the group is first tested as one span.
    # If it is not spacing-only, each op is judged on its own, so one error
    # (세 개 -> 3개) cannot hide its neighbour (미군 -> 미국).
    groups, cur = [], []
    for c in out.alignments[0]:
        if c.type == "equal":
            if cur:
                groups.append(cur)
                cur = []
            continue
        ra, ha = r[c.ref_start_idx:c.ref_end_idx], h[c.hyp_start_idx:c.hyp_end_idx]
        if c.type == "substitute":
            cur.extend(([x], [y]) for x, y in zip(ra, ha))  # substitutions are 1:1 per word
        else:
            cur.append((ra, ha))
    if cur:
        groups.append(cur)
    chunks = []
    for g in groups:
        gr = [w for a, _ in g for w in a]
        gh = [w for _, b in g for w in b]
        if nospace(" ".join(gr)) == nospace(" ".join(gh)) or len(g) == 1 or _native_only(gr, gh):
            chunks.append(Chunk(" ".join(gr), " ".join(gh), categorize(gr, gh)))
        else:
            chunks.extend(Chunk(" ".join(a), " ".join(b), categorize(a, b)) for a, b in g)
    return chunks


def utterance_label(ref: str, hyp: str) -> dict:
    """Worst severity in the utterance, plus a length check for ref/audio mismatch."""
    chunks = classify(ref, hyp)
    worst = max((RANK[c.severity] for c in chunks), default=-1)
    rl, hl = len(nospace(apply_upto(ref, 4))), len(nospace(apply_upto(hyp, 4)))
    ratio = hl / rl if rl else float("inf")
    return {
        "severity": {-1: "clean", 0: "format", 1: "ambiguous", 2: "meaning"}[worst],
        "length_ratio": round(ratio, 3),
        # A transcript much longer or shorter than its reference is more often a
        # reference/audio mismatch than a recognition error. Needs a human listen.
        "suspect_ref_audio_mismatch": ratio > 1.5 or ratio < 0.6,
        "chunks": [{"ref": c.ref, "hyp": c.hyp, "category": c.category, "severity": c.severity} for c in chunks],
    }


def main(argv=None) -> None:
    """Label every utterance in a run: python -m koasr.classify runs/<run_dir>"""
    import argparse
    import json
    from collections import Counter
    from pathlib import Path

    p = argparse.ArgumentParser()
    p.add_argument("run_dir")
    a = p.parse_args(argv)
    d = Path(a.run_dir)
    rows = [json.loads(l) for l in (d / "hyp.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    sev, cat, flagged = Counter(), Counter(), []
    with open(d / "classified.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            lab = utterance_label(r["ref"], r["hyp"])
            sev[lab["severity"]] += 1
            cat.update(c["category"] for c in lab["chunks"])
            if lab["suspect_ref_audio_mismatch"]:
                flagged.append(r["uid"])
            f.write(json.dumps({"uid": r["uid"], **lab}, ensure_ascii=False) + "\n")
    summary = {"utterances": len(rows), "by_worst_severity": dict(sev),
               "chunks_by_category": dict(cat.most_common()), "suspect_ref_audio_mismatch": flagged}
    (d / "classified_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
