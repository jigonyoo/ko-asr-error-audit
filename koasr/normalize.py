"""Normalization layers for Korean ASR scoring, applied one at a time.

Each layer removes one kind of difference that is about *writing* rather than
*hearing*. M2 measures how much CER each layer removes, so a reader can see
how much of a "word error rate" was ever an error.

Layers, in the cumulative order used by the ablation:
  nfc         Unicode NFC (composed Hangul). NFD jamo silently breaks string equality.
  paren_latin drop parentheticals that contain Latin script, e.g. 깁슨(Gibson):
              written into the reference, never spoken.
  punct       drop punctuation, but keep '.' and ',' between digits and '%'.
  numerals    read Arabic numerals as Sino-Korean words (2005 -> 이천오, 51% -> 오십일퍼센트).
  nospace     remove all whitespace (spacing-insensitive).

Known limits, kept visible on purpose:
  - native Korean numerals (스물, 아홉 시, 세 개) are not produced; 20 -> 이십 still
    mismatches a reference that says 스물.
  - '쩜' (colloquial decimal point) is not mapped to '점'.
  - English words vs Hangul transliteration (nvidia vs 엔비디아) are untouched.
"""
from __future__ import annotations

import re
import unicodedata

DIGITS = "영일이삼사오육칠팔구"
SMALL_UNITS = [(1000, "천"), (100, "백"), (10, "십")]
BIG_UNITS = [(10**12, "조"), (10**8, "억"), (10**4, "만")]
UNITS = [("%", "퍼센트"), ("km", "킬로미터"), ("cm", "센티미터"), ("mm", "밀리미터"),
         ("kg", "킬로그램")]


def _below_10000(n: int) -> str:
    out = []
    for value, name in SMALL_UNITS:
        d, n = divmod(n, value)
        if d:
            out.append(("" if d == 1 else DIGITS[d]) + name)
    if n:
        out.append(DIGITS[n])
    return "".join(out)


def read_sino(n: int) -> str:
    """Sino-Korean reading of a non-negative integer (12919 -> 만이천구백십구)."""
    if n == 0:
        return "영"
    out = []
    for value, name in BIG_UNITS:
        d, n = divmod(n, value)
        if d:
            head = _below_10000(d)
            if name == "만" and d == 1:
                head = ""  # 만, not 일만; but 일억, 일조
            out.append(head + name)
    if n:
        out.append(_below_10000(n))
    return "".join(out)


def read_number_token(tok: str) -> str:
    """'2,919' -> 이천구백십구, '5.94' -> 오점구사 (digits after the point read one by one)."""
    tok = re.sub(r"(?<=\d),(?=\d{3}\b)", "", tok)
    if "." in tok:
        whole, frac = tok.split(".", 1)
        return read_sino(int(whole)) + "점" + "".join(DIGITS[int(c)] for c in frac)
    return read_sino(int(tok))


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def paren_latin(s: str) -> str:
    return re.sub(r"\s*\([^)]*[A-Za-z][^)]*\)", "", s)


def punct(s: str) -> str:
    keep_between_digits = re.sub(r"(?<=\d)([.,])(?=\d)", "\x00\\1", s)
    chars = []
    for i, ch in enumerate(keep_between_digits):
        if ch == "\x00":
            continue
        prev = keep_between_digits[i - 1] if i else ""
        if prev == "\x00" or ch == "%":
            chars.append(ch)
        elif unicodedata.category(ch).startswith("P"):
            chars.append(" ")
        else:
            chars.append(ch)
    return re.sub(r"\s+", " ", "".join(chars)).strip()


def numerals(s: str) -> str:
    s = re.sub(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?", lambda m: read_number_token(m.group(0)), s)
    for sym, word in UNITS:
        s = re.sub(rf"(?<=[{DIGITS}십백천만억조점])\s*{re.escape(sym)}(?![A-Za-z])", word, s)
    return s


def nospace(s: str) -> str:
    return re.sub(r"\s+", "", s)


LAYERS = [("nfc", nfc), ("paren_latin", paren_latin), ("punct", punct),
          ("numerals", numerals), ("nospace", nospace)]


def apply_upto(s: str, k: int) -> str:
    """Apply the first k layers in order (k=0 is the raw string)."""
    for _, fn in LAYERS[:k]:
        s = fn(s)
    return s
