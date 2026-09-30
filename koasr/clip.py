"""Export one utterance's audio to a local WAV so a person can listen to it.

    python -m koasr.clip --dataset zeroth --uid zeroth-0001-105_003_0478

WAV files are git-ignored; the audio stays on this machine.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import soundfile as sf

from .data import TARGET_SR, load_utterances


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--uid", required=True, help="uid as printed in hyp.jsonl / worst10.md")
    p.add_argument("--out", default="clips")
    a = p.parse_args(argv)
    for u in load_utterances(a.dataset, None):
        if u.uid == a.uid or u.uid.endswith("-" + a.uid.split("-", 1)[-1]):
            out = Path(a.out); out.mkdir(exist_ok=True)
            path = out / f"{u.uid}.wav"
            sf.write(path, u.audio, TARGET_SR)
            print(f"{path}\nREF: {u.ref}")
            return
    raise SystemExit(f"uid not found: {a.uid}")


if __name__ == "__main__":
    main()
