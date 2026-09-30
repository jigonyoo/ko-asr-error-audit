from koasr.classify import classify, utterance_label


def cats(ref, hyp):
    return [c.category for c in classify(ref, hyp)]


def test_clean_and_format_only():
    assert cats("오늘 날씨", "오늘 날씨") == []
    assert cats("학교에 간다", "학교에간다") == ["spacing"]
    assert cats("미국인의 오십 일 퍼센트는", "미국인의 51%는") == ["spacing"]


def test_inserted_negation_is_meaning():
    assert cats("나는 학교에 간다", "나는 학교에 안 간다") == ["negation"]


def test_man_i_is_not_a_number():
    # 만이 ('only') must not be read as the numeral 만.
    assert "number" not in cats("돌연변이만이 유전될", "도련변이 많이 유전될")


def test_missing_eok_is_a_number_error():
    assert "number" in cats("매출 일 조 이천 구백 십 구 억원", "매출 1조이 2,919원")


def test_native_vs_sino_numeral_is_flagged_not_punished():
    assert cats("두께 스물 센티미터를", "두께 20cm 를") == ["native_numeral"]


def test_adjacent_errors_do_not_hide_each_other():
    c = cats("미군은 한반도에 세 개", "미국은 한반도의 3개")
    assert "sound_alike" in c and "particle" in c


def test_length_ratio_flags_reference_audio_mismatch():
    lab = utterance_label("반면 하종강 교수는 인간에 대한 이해 부족이 낳은 현상이라며 강조했다", "반면 하중간 교수는")
    assert lab["suspect_ref_audio_mismatch"]
