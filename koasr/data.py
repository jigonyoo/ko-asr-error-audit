"""Test-split loaders for the two public Korean corpora.

Both are CC BY 4.0. Audio is downloaded to the Hugging Face cache as parquet
and decoded in memory; it is never written into this repository.

We read parquet directly with pyarrow instead of the `datasets` library:
`datasets` 3.x pins dill 0.3.8, whose pickler breaks on Python 3.14
(`Pickler._batch_setitems() takes 2 positional arguments but 3 were given`).
"""
from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Iterable, Iterator

import numpy as np

TARGET_SR = 16_000
PARQUET_BRANCH = "refs/convert/parquet"  # the Hub's auto-converted copy

DATASETS = {
    "zeroth": {
        "hub_id": "kresnik/zeroth_korean",
        "config": None,
        "split": "test",
        "ref_fields": ["text", "transcript", "sentence", "transcription"],
        "license": "CC BY 4.0",
        "source": "https://www.openslr.org/40/",
    },
    "fleurs": {
        "hub_id": "google/fleurs",
        "config": "ko_kr",
        "split": "test",
        # raw_transcription is the un-normalized original. `transcription` is
        # FLEURS's own normalized version; mixing the two silently shrinks CER.
        "ref_fields": ["raw_transcription"],
        "license": "CC BY 4.0",
        "source": "https://huggingface.co/datasets/google/fleurs",
    },
}


@dataclass
class Utterance:
    uid: str
    audio: np.ndarray  # float32 mono at TARGET_SR
    ref: str
    ref_field: str


def select_parquet(files: Iterable[str], config: str | None, split: str) -> list[str]:
    """Pick the parquet shards for one config/split from a repo file listing.

    Handles both layouts seen on the Hub:
      data/test-00000-of-00001.parquet            (native parquet repos)
      ko_kr/test/0000.parquet, default/test/...   (refs/convert/parquet)
    """
    out = []
    for f in files:
        if not f.endswith(".parquet"):
            continue
        parts = f.split("/")
        name = parts[-1]
        in_split = (split in parts[:-1]) or name.startswith(f"{split}-") or name.startswith(f"{split}.")
        if not in_split:
            continue
        if config and config not in parts[:-1]:
            continue
        out.append(f)
    return sorted(out)


def resolve_files(hub_id: str, config: str | None, split: str) -> tuple[list[str], str]:
    """Download the needed shards; return (local paths, revision used)."""
    from huggingface_hub import hf_hub_download, list_repo_files

    for revision in ("main", PARQUET_BRANCH):
        try:
            listing = list_repo_files(hub_id, repo_type="dataset", revision=revision)
        except Exception:
            continue
        chosen = select_parquet(listing, config, split)
        if chosen:
            paths = [
                hf_hub_download(hub_id, f, repo_type="dataset", revision=revision)
                for f in chosen
            ]
            return paths, revision
    raise FileNotFoundError(
        f"no parquet shards for {hub_id} config={config} split={split} "
        f"on 'main' or '{PARQUET_BRANCH}'"
    )


def _decode(audio_cell) -> tuple[np.ndarray, int]:
    import soundfile as sf

    if isinstance(audio_cell, dict) and audio_cell.get("bytes"):
        data, sr = sf.read(io.BytesIO(audio_cell["bytes"]), dtype="float32", always_2d=False)
    elif isinstance(audio_cell, dict) and "array" in audio_cell:
        data, sr = np.asarray(audio_cell["array"], dtype=np.float32), audio_cell["sampling_rate"]
    else:
        raise ValueError(f"unsupported audio cell: {type(audio_cell).__name__}")
    if data.ndim > 1:  # downmix; the corpora are mono, so this should not trigger
        data = data.mean(axis=1).astype(np.float32)
    return data, sr


def _pick_ref(row: dict, fields: list[str]) -> tuple[str, str]:
    for f in fields:
        if isinstance(row.get(f), str):
            return row[f], f
    raise KeyError(f"none of the reference fields {fields} found; columns: {sorted(row)}")


def iter_parquet(paths: list[str], name: str, ref_fields: list[str], limit: int | None) -> Iterator[Utterance]:
    """Yield utterances from local parquet shards, in file order."""
    import pyarrow.parquet as pq

    n = 0
    for path in paths:
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=32):
            for row in batch.to_pylist():
                if limit is not None and n >= limit:
                    return
                ref, field = _pick_ref(row, ref_fields)
                audio, sr = _decode(row["audio"])
                if sr != TARGET_SR:
                    raise ValueError(f"{name} row {n}: sampling rate {sr}, expected {TARGET_SR}")
                # FLEURS "id" is the sentence id, and several speakers read the
                # same sentence, so it is not unique. Prefix the row position.
                src_id = row.get("id", row.get("path", ""))
                yield Utterance(uid=f"{name}-{n:04d}-{src_id}", audio=audio, ref=ref, ref_field=field)
                n += 1


def load_utterances(name: str, limit: int | None = None) -> Iterator[Utterance]:
    spec = DATASETS[name]
    paths, revision = resolve_files(spec["hub_id"], spec["config"], spec["split"])
    spec["revision_used"] = revision  # recorded in the run manifest
    yield from iter_parquet(paths, name, spec["ref_fields"], limit)
