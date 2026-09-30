import io

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import soundfile as sf

from koasr.data import TARGET_SR, iter_parquet, select_parquet


def test_select_native_layout():
    files = ["README.md", "data/train-00000-of-00003.parquet",
             "data/test-00000-of-00001.parquet"]
    assert select_parquet(files, None, "test") == ["data/test-00000-of-00001.parquet"]


def test_select_converted_layout_respects_config():
    files = ["ko_kr/test/0000.parquet", "ko_kr/train/0000.parquet",
             "ja_jp/test/0000.parquet", "ko_kr/validation/0000.parquet"]
    assert select_parquet(files, "ko_kr", "test") == ["ko_kr/test/0000.parquet"]


def test_select_does_not_match_substring_of_other_split():
    # "test" must not match a file that merely contains the word, e.g. "latest".
    files = ["data/latest-00000.parquet", "data/test-00000.parquet"]
    assert select_parquet(files, None, "test") == ["data/test-00000.parquet"]


def _wav_bytes(seconds: float, sr: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, np.zeros(int(seconds * sr), dtype=np.float32), sr, format="WAV")
    return buf.getvalue()


def _write(tmp_path, rows):
    table = pa.Table.from_pylist(rows)
    path = tmp_path / "t.parquet"
    pq.write_table(table, path)
    return [str(path)]


def test_iter_parquet_decodes_audio_and_picks_ref(tmp_path):
    rows = [{"id": i, "text": f"문장 {i}", "audio": {"bytes": _wav_bytes(0.5, TARGET_SR), "path": f"{i}.wav"}}
            for i in range(3)]
    utts = list(iter_parquet(_write(tmp_path, rows), "zeroth", ["text"], limit=2))
    assert [u.uid for u in utts] == ["zeroth-0000-0", "zeroth-0001-1"]
    assert utts[0].ref == "문장 0" and utts[0].ref_field == "text"
    assert utts[0].audio.dtype == np.float32 and len(utts[0].audio) == TARGET_SR // 2


def test_iter_parquet_refuses_wrong_sampling_rate(tmp_path):
    rows = [{"id": 0, "text": "가", "audio": {"bytes": _wav_bytes(0.1, 8_000), "path": "0.wav"}}]
    with pytest.raises(ValueError, match="sampling rate 8000"):
        list(iter_parquet(_write(tmp_path, rows), "zeroth", ["text"], limit=None))


def test_iter_parquet_prefers_raw_transcription_for_fleurs(tmp_path):
    rows = [{"id": 7, "transcription": "정규화된 문장", "raw_transcription": "정규화된 문장.",
             "audio": {"bytes": _wav_bytes(0.1, TARGET_SR), "path": "7.wav"}}]
    u = next(iter_parquet(_write(tmp_path, rows), "fleurs", ["raw_transcription"], limit=None))
    assert u.ref == "정규화된 문장." and u.ref_field == "raw_transcription"


def test_uid_is_unique_when_source_id_repeats(tmp_path):
    # FLEURS: two speakers reading sentence 1907 share id 1907.
    rows = [{"id": 1907, "raw_transcription": "같은 문장", "audio": {"bytes": _wav_bytes(0.1, TARGET_SR), "path": f"{i}.wav"}}
            for i in range(2)]
    uids = [u.uid for u in iter_parquet(_write(tmp_path, rows), "fleurs", ["raw_transcription"], limit=None)]
    assert len(set(uids)) == 2
