"""Compare the classifier with human judgments (annotations/judgments.jsonl).

    python -m koasr.agreement annotations/judgments.jsonl runs/<zeroth_run> runs/<fleurs_run>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .classify import RANK, classify, utterance_label

HUMAN_TO_SEVERITY = {"meaning_changed": "meaning", "format_only": "format",
                     "format_mismatch_ambiguous": "ambiguous"}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("judgments")
    p.add_argument("run_dirs", nargs="+")
    p.add_argument("--out", default="runs/agreement.json")
    a = p.parse_args(argv)
    rows = {}
    for d in a.run_dirs:
        for l in (Path(d) / "hyp.jsonl").read_text(encoding="utf-8").splitlines():
            r = json.loads(l)
            rows[r["uid"].split("-")[-1]] = r
    out = []
    for l in Path(a.judgments).read_text(encoding="utf-8").splitlines():
        j = json.loads(l)
        if "ref_span" in j:
            chunks = classify(j["ref_span"], j["hyp_span"])
            worst = max(chunks, key=lambda c: RANK[c.severity]) if chunks else None
            machine = worst.severity if worst else "clean"
            human = HUMAN_TO_SEVERITY[j["judgment"]]
            out.append({"kind": "span", "ref": j["ref_span"], "hyp": j["hyp_span"], "human": j["judgment"],
                        "machine": machine, "machine_category": worst.category if worst else None,
                        "agree": machine == human})
        else:
            r = rows.get(j["uid"].split("-")[-1])
            if r is None:
                continue
            lab = utterance_label(r["ref"], r["hyp"])
            if j["judgment"] == "reference_audio_mismatch":
                agree = lab["suspect_ref_audio_mismatch"]
                machine = f"length_ratio={lab['length_ratio']} flag={agree}"
            else:  # parenthetical_spoken: the machine cannot know what was said
                agree = None
                machine = "needs audio; not decidable from text"
            out.append({"kind": "utterance", "uid": j["uid"], "human": j["judgment"],
                        "machine": machine, "agree": agree})
    decided = [x for x in out if x["agree"] is not None]
    summary = {"n": len(out), "decidable": len(decided), "agree": sum(bool(x["agree"]) for x in decided)}
    Path(a.out).write_text(json.dumps({"summary": summary, "items": out}, ensure_ascii=False, indent=2), encoding="utf-8")
    for x in out:
        mark = {True: "AGREE", False: "DISAGREE", None: "n/a"}[x["agree"]]
        print(f"{mark:8s} human={x['human']:26s} machine={x['machine']}  {x.get('machine_category') or ''}  {x.get('ref', x.get('uid'))}")
    print(summary)


if __name__ == "__main__":
    main()
