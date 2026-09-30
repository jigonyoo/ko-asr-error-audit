"""M1 runner: transcribe a test split, write per-utterance rows + corpus metrics.

Outputs go to runs/<UTC timestamp>_<dataset>_<model>_n<N>/:
  hyp.jsonl      one row per utterance (uid, ref, hyp, cer, wer)
  metrics.json   pooled CER/WER over the run
  manifest.json  everything needed to reproduce the run
  worst10.md     the ten highest-CER utterances, ref and hyp side by side
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from . import metrics
from .data import DATASETS, load_utterances


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return None


def run(dataset: str, utterances, transcriber, out_root: Path, limit, describe: dict) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    rows, refs, hyps = [], [], []
    t0 = time.time()
    for u in utterances:
        hyp = transcriber(u.audio)
        rows.append({
            "uid": u.uid,
            "ref_field": u.ref_field,
            "ref": u.ref,
            "hyp": hyp,
            "cer": metrics.cer(u.ref, hyp),
            "wer": metrics.wer(u.ref, hyp),
            "audio_seconds": round(len(u.audio) / 16_000, 2),
        })
        refs.append(u.ref)
        hyps.append(hyp)
        print(f"[{len(rows)}] cer={rows[-1]['cer']:.3f}  {u.uid}", flush=True)
    elapsed = time.time() - t0
    if not rows:
        raise RuntimeError("no utterances were processed")

    model_tag = describe.get("model", "model").replace("/", "-")
    out = out_root / f"{stamp}_{dataset}_{model_tag}_n{len(rows)}"
    out.mkdir(parents=True, exist_ok=False)

    with open(out / "hyp.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    m = metrics.corpus_rates(refs, hyps)
    m.update({
        "normalization": "none (raw strings, M1 baseline)",
        "audio_seconds_total": round(sum(r["audio_seconds"] for r in rows), 1),
        "wall_seconds": round(elapsed, 1),
    })
    (out / "metrics.json").write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")

    spec = DATASETS.get(dataset, {})
    manifest = {
        "created_utc": stamp,
        "dataset": dataset,
        "hub_id": spec.get("hub_id"),
        "config": spec.get("config"),
        "split": spec.get("split"),
        "revision_used": spec.get("revision_used"),
        "license": spec.get("license"),
        "source": spec.get("source"),
        "limit": limit,
        "n_processed": len(rows),
        "transcriber": describe,
        "git_commit": _git_commit(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    worst = sorted(rows, key=lambda r: r["cer"], reverse=True)[:10]
    lines = [f"# Worst 10 by CER — {dataset} / {model_tag}\n",
             "| # | CER | uid | reference | whisper |", "|---|---|---|---|---|"]
    for i, r in enumerate(worst, 1):
        ref = r["ref"].replace("|", "\\|")
        hyp = r["hyp"].replace("|", "\\|")
        lines.append(f"| {i} | {r['cer']:.3f} | `{r['uid']}` | {ref} | {hyp} |")
    (out / "worst10.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\nCER {m['cer']:.4f}  WER {m['wer']:.4f}  n={m['n_utterances']}  -> {out}")
    return out


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="M1 baseline: raw CER/WER for Korean ASR")
    p.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    p.add_argument("--model", default="small", help="faster-whisper model, e.g. small, large-v3-turbo")
    p.add_argument("--limit", type=int, default=50, help="utterances to process (0 = whole split)")
    p.add_argument("--out", default="runs")
    a = p.parse_args(argv)

    from .transcribe import WhisperTranscriber

    limit = None if a.limit == 0 else a.limit
    tr = WhisperTranscriber(a.model)
    run(a.dataset, load_utterances(a.dataset, limit), tr, Path(a.out), limit, tr.describe())


if __name__ == "__main__":
    main()
