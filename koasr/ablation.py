"""M2: how much of the raw error rate is writing, not hearing.

Reads a run's hyp.jsonl (no re-transcription) and reports pooled CER/WER after
each cumulative normalization layer, plus the ten utterances that stay worst
after every layer: those are the candidates for human judgment in M3.

    python -m koasr.ablation runs/<run_dir>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import jiwer

from .normalize import LAYERS, apply_upto


def ablate(rows: list[dict]) -> dict:
    table = []
    for k in range(len(LAYERS) + 1):
        refs = [apply_upto(r["ref"], k) for r in rows]
        hyps = [apply_upto(r["hyp"], k) for r in rows]
        keep = [i for i, x in enumerate(refs) if x]
        refs = [refs[i] for i in keep]
        hyps = [hyps[i] for i in keep]
        out = jiwer.process_characters(refs, hyps)
        entry = {
            "layer": "raw" if k == 0 else LAYERS[k - 1][0],
            "cer": out.cer,
            # Absolute character edits. CER's denominator shrinks as layers delete
            # spaces and punctuation from the reference, so CER alone understates
            # how many edits a layer removed.
            "char_edits": out.substitutions + out.deletions + out.insertions,
            "ref_chars": sum(len(x) for x in refs),
            # WER is undefined once spaces are gone.
            "wer": None if k == len(LAYERS) else jiwer.wer(refs, hyps),
            "n": len(refs),
        }
        table.append(entry)
    for prev, cur in zip(table, table[1:]):
        cur["cer_delta"] = cur["cer"] - prev["cer"]

    k = len(LAYERS)
    residual = []
    for r in rows:
        ref, hyp = apply_upto(r["ref"], k), apply_upto(r["hyp"], k)
        if ref:
            residual.append({"uid": r["uid"], "cer_raw": r["cer"], "cer_norm": jiwer.cer(ref, hyp),
                             "ref": r["ref"], "hyp": r["hyp"]})
    residual.sort(key=lambda x: x["cer_norm"], reverse=True)
    return {"layers": table, "residual_worst10": residual[:10]}


def to_markdown(res: dict, title: str) -> str:
    lines = [f"# Normalization ablation — {title}\n",
             "| layer (cumulative) | CER | ΔCER | char edits | ref chars | WER |", "|---|---|---|---|---|---|"]
    for e in res["layers"]:
        d = f"{e['cer_delta']:+.4f}" if "cer_delta" in e else ""
        w = f"{e['wer']:.4f}" if e["wer"] is not None else "n/a"
        lines.append(f"| {e['layer']} | {e['cer']:.4f} | {d} | {e['char_edits']} | {e['ref_chars']} | {w} |")
    lines += ["", "## Worst 10 after all layers (candidates for human judgment)\n",
              "| uid | CER raw | CER norm | reference | whisper |", "|---|---|---|---|---|"]
    for r in res["residual_worst10"]:
        ref, hyp = r["ref"].replace("|", "\\|"), r["hyp"].replace("|", "\\|")
        lines.append(f"| `{r['uid']}` | {r['cer_raw']:.3f} | {r['cer_norm']:.3f} | {ref} | {hyp} |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("run_dir")
    a = p.parse_args(argv)
    d = Path(a.run_dir)
    rows = [json.loads(l) for l in (d / "hyp.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    res = ablate(rows)
    (d / "ablation.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    (d / "ablation.md").write_text(to_markdown(res, d.name), encoding="utf-8")
    print(to_markdown(res, d.name))


if __name__ == "__main__":
    main()
