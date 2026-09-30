"""Raw (un-normalized) CER / WER for Korean.

M1 deliberately applies NO normalization: no spacing removal, no NFC, no
punctuation stripping, no number conversion. Each of those becomes a separate,
measured layer in M2, so its effect shows up as a number instead of being
silently folded into the baseline.
"""
from __future__ import annotations

import jiwer


def _guard(ref: str) -> None:
    if not ref:
        # jiwer raises on an empty reference; we want the caller to see why.
        raise ValueError("empty reference transcript: CER/WER undefined")


def cer(ref: str, hyp: str) -> float:
    """Character error rate over the raw strings (spaces count as characters)."""
    _guard(ref)
    return jiwer.cer(ref, hyp)


def wer(ref: str, hyp: str) -> float:
    """Word error rate, words split on whitespace (Korean 어절)."""
    _guard(ref)
    return jiwer.wer(ref, hyp)


def corpus_rates(refs: list[str], hyps: list[str]) -> dict:
    """Corpus-level rates: total edits / total reference units.

    Averaging per-utterance CERs would overweight short sentences; jiwer on the
    full lists gives the pooled rate, which is the number papers report.
    """
    if len(refs) != len(hyps):
        raise ValueError(f"length mismatch: {len(refs)} refs vs {len(hyps)} hyps")
    for r in refs:
        _guard(r)
    return {
        "cer": jiwer.cer(refs, hyps),
        "wer": jiwer.wer(refs, hyps),
        "n_utterances": len(refs),
        "ref_chars": sum(len(r) for r in refs),
        "ref_words": sum(len(r.split()) for r in refs),
    }
