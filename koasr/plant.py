"""Planted-defect validation for the M3 classifier.

Take clean reference sentences, plant one known defect per copy, and check
whether the classifier (a) notices a difference and (b) calls it the right
severity. Controls with format-only changes measure false alarms.

    python -m koasr.plant runs/<zeroth_run> runs/<fleurs_run> [--exclude annotations/judgments.jsonl]
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from .classify import classify

SWAP = {"은": "이", "는": "가", "을": "은", "를": "는", "에": "의", "의": "에"}


def plant_negation(s: str):
    w = s.split()
    for i in range(len(w) - 1, -1, -1):
        if re.search(r"(다|요|니다)[.?!]?$", w[i]):
            return " ".join(w[:i] + ["안"] + w[i:])
    return None


def plant_number(s: str):
    m = re.search(r"\d", s)
    if m:
        d = s[m.start()]
        return s[:m.start()] + str((int(d) + 3) % 10 or 7) + s[m.end():]
    m = re.search(r"(이|삼|사|오|육|칠|팔|구)(십|백|천|만|억)", s)
    if m:
        new = {"이": "삼", "삼": "사", "사": "오", "오": "육", "육": "칠", "칠": "팔", "팔": "구", "구": "이"}[m.group(1)]
        return s[:m.start()] + new + s[m.start() + 1:]
    return None


def plant_particle(s: str):
    w = s.split()
    for i, x in enumerate(w):
        core = x.rstrip(".,?!")
        if len(core) >= 3 and core[-1] in SWAP:
            w[i] = core[:-1] + SWAP[core[-1]] + x[len(core):]
            return " ".join(w)
    return None


def plant_sound_alike(s: str):
    # Drop the final consonant of one syllable in a 3+ syllable word: 곤충 -> 고충.
    w = s.split()
    for i, x in enumerate(w):
        if len(x) >= 3:
            for j, ch in enumerate(x[:-1]):
                code = ord(ch) - 0xAC00
                if 0 <= code < 11172 and code % 28:
                    w[i] = x[:j] + chr(ord(ch) - code % 28) + x[j + 1:]
                    return " ".join(w)
    return None


def plant_omission(s: str):
    w = s.split()
    return " ".join(w[: len(w) // 2]) if len(w) >= 8 else None


def control_spacing(s: str):
    return s.replace(" ", "", 1) if " " in s else None


def control_punct(s: str):
    return s.rstrip(".") + "."


PLANTS = {
    "negation": (plant_negation, "meaning"),
    "number": (plant_number, "meaning"),
    "particle": (plant_particle, "ambiguous_or_meaning"),
    "sound_alike": (plant_sound_alike, "meaning"),
    "omission": (plant_omission, "meaning"),
}
CONTROLS = {"spacing": control_spacing, "punct": control_punct}


def run(refs: list[str]) -> dict:
    res = {"planted": {}, "controls": {}}
    for name, (fn, expect) in PLANTS.items():
        n = det = sev_ok = 0
        cats = Counter()
        misses = []
        for r in refs:
            c = fn(r)
            if c is None or c == r:
                continue
            n += 1
            chunks = classify(r, c)
            if chunks:
                det += 1
            sevs = {x.severity for x in chunks}
            cats.update(x.category for x in chunks)
            ok = ("meaning" in sevs) if expect == "meaning" else bool(sevs & {"ambiguous", "meaning"})
            sev_ok += ok
            if not ok and len(misses) < 5:
                misses.append({"ref": r, "planted": c, "got": [(x.category, x.ref, x.hyp) for x in chunks]})
        res["planted"][name] = {"n": n, "detected": det, "severity_ok": sev_ok,
                                "categories": dict(cats), "misses": misses}
    for name, fn in CONTROLS.items():
        n = alarms = 0
        examples = []
        for r in refs:
            c = fn(r)
            if c is None or c == r:
                continue
            n += 1
            bad = [x for x in classify(r, c) if x.severity != "format"]
            if bad:
                alarms += 1
                if len(examples) < 5:
                    examples.append({"ref": r, "control": c, "got": [(x.category, x.ref, x.hyp) for x in bad]})
        res["controls"][name] = {"n": n, "false_alarms": alarms, "examples": examples}
    return res


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("run_dirs", nargs="+")
    p.add_argument("--exclude", help="judgments.jsonl; rows judged reference_audio_mismatch are skipped")
    p.add_argument("--out", default="runs/plant_validation.json")
    a = p.parse_args(argv)
    skip = set()
    if a.exclude:
        for l in Path(a.exclude).read_text(encoding="utf-8").splitlines():
            j = json.loads(l)
            if j.get("judgment") == "reference_audio_mismatch":
                skip.add(j["uid"].split("-", 1)[-1])
    refs = []
    for d in a.run_dirs:
        for l in (Path(d) / "hyp.jsonl").read_text(encoding="utf-8").splitlines():
            row = json.loads(l)
            if row["uid"].split("-", 1)[-1].split("-")[-1] in {s.split("-")[-1] for s in skip}:
                continue
            refs.append(row["ref"])
    res = run(refs)
    res["n_refs"] = len(refs)
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    for k, v in res["planted"].items():
        print(f"planted {k:12s} n={v['n']:3d} detected={v['detected']:3d} severity_ok={v['severity_ok']:3d}  {v['categories']}")
    for k, v in res["controls"].items():
        print(f"control {k:12s} n={v['n']:3d} false_alarms={v['false_alarms']}")


if __name__ == "__main__":
    main()
