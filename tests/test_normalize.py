import unicodedata

import pytest

from koasr.normalize import apply_upto, nfc, numerals, paren_latin, punct, read_number_token, read_sino


@pytest.mark.parametrize("n,expected", [
    (0, "영"), (10, "십"), (11, "십일"), (19, "십구"), (51, "오십일"), (100, "백"),
    (1000, "천"), (2005, "이천오"), (10000, "만"), (12919, "만이천구백십구"),
    (10**8, "일억"), (10**12, "일조"), (1_2919_0000_0000, "일조이천구백십구억"),
])
def test_read_sino(n, expected):
    assert read_sino(n) == expected


def test_number_tokens():
    assert read_number_token("2,919") == "이천구백십구"
    assert read_number_token("5.94") == "오점구사"


def test_comma_between_two_digit_numbers_is_a_list_not_thousands():
    # "17,18대" (17th and 18th) must not become 1718.
    assert numerals("17,18대") == "십칠,십팔대"


def test_units_follow_numbers_only():
    assert numerals("51%") == "오십일퍼센트"
    assert numerals("20cm") == "이십센티미터"
    assert numerals("cm 단위") == "cm 단위"


def test_nfc_joins_decomposed_jamo():
    decomposed = unicodedata.normalize("NFD", "한국어")
    assert decomposed != "한국어" and nfc(decomposed) == "한국어"


def test_paren_latin_drops_romanization_only():
    assert paren_latin("깁슨(Gibson)은") == "깁슨은"
    assert paren_latin("(주)삼성") == "(주)삼성"


def test_punct_keeps_decimals_and_percent():
    assert punct('"1940년 8월 15일, 연합군은 ""드래군 작전""이라 불렸다. "') == "1940년 8월 15일 연합군은 드래군 작전 이라 불렸다"
    assert punct("5.94시간, 51%.") == "5.94시간 51%"


def test_full_pipeline_matches_hangul_and_digit_forms():
    # Zeroth writes numbers in Hangul with spaces; Whisper writes digits.
    ref = "미국인의 오십 일 퍼센트는"
    hyp = "미국인의 51%는"
    assert apply_upto(ref, 5) == apply_upto(hyp, 5)


def test_native_numeral_mismatch_is_left_visible():
    # 스물 (native) vs 20 -> 이십 (Sino): deliberately not "fixed".
    assert apply_upto("스물 센티미터", 5) != apply_upto("20cm", 5)
