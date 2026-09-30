import json
from pathlib import Path

import numpy as np
import pytest

from koasr import metrics
from koasr.data import Utterance
from koasr.run import run


def test_negation_insertion_is_small_cer_but_flips_meaning():
    # "나는 학교에 간다" = 9 characters including 2 spaces.
    # Inserting "안 " is 2 character edits -> CER 2/9, yet the meaning is inverted.
    # This is the case the whole project exists to surface.
    assert metrics.cer("나는 학교에 간다", "나는 학교에 안 간다") == pytest.approx(2 / 9)
    assert metrics.wer("나는 학교에 간다", "나는 학교에 안 간다") == pytest.approx(1 / 3)


def test_spacing_only_error_inflates_wer_not_cer():
    # One missing space: CER 1/6, but WER counts both 어절 as wrong -> 2/2.
    assert metrics.cer("학교에 간다", "학교에간다") == pytest.approx(1 / 6)
    assert metrics.wer("학교에 간다", "학교에간다") == pytest.approx(1.0)


def test_identical_is_zero():
    assert metrics.cer("오늘 날씨", "오늘 날씨") == 0.0


def test_empty_reference_is_rejected():
    with pytest.raises(ValueError):
        metrics.cer("", "뭔가")


def test_corpus_rate_is_pooled_not_averaged():
    # Short utterance fully wrong (1/1) + long one right (0/10):
    # averaging per-utterance CER gives 0.5, pooling gives 1/11.
    refs = ["가", "가나다라마바사아자차"]
    hyps = ["나", "가나다라마바사아자차"]
    assert metrics.corpus_rates(refs, hyps)["cer"] == pytest.approx(1 / 11)


def test_runner_writes_all_artifacts(tmp_path: Path):
    utts = [
        Utterance("t-1", np.zeros(16_000, dtype=np.float32), "나는 학교에 간다", "text"),
        Utterance("t-2", np.zeros(8_000, dtype=np.float32), "오늘 날씨", "text"),
    ]
    fake = {"t-1": "나는 학교에 안 간다", "t-2": "오늘 날씨"}
    it = iter(utts)
    current = {}

    def gen():
        for u in it:
            current["uid"] = u.uid
            yield u

    out = run("zeroth", gen(), lambda audio: fake[current["uid"]], tmp_path, 2, {"model": "fake"})
    rows = [json.loads(l) for l in (out / "hyp.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["uid"] for r in rows] == ["t-1", "t-2"]
    m = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert m["n_utterances"] == 2
    assert m["cer"] == pytest.approx(2 / 14)  # 2 edits over 9 + 5 reference chars
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["license"] == "CC BY 4.0"
    assert (out / "worst10.md").read_text(encoding="utf-8").count("나는 학교에 안 간다") == 1
