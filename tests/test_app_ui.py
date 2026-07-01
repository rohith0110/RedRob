from app import should_run_ranking


def test_ranking_starts_only_after_button_click():
    assert not should_run_ranking(has_upload=False, run_clicked=False)
    assert not should_run_ranking(has_upload=True, run_clicked=False)
    assert should_run_ranking(has_upload=True, run_clicked=True)
