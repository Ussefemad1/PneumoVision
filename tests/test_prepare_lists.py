"""tools/pv/prepare_lists.py on tiny SYNTHETIC listfiles (no MIMIC data)."""

from __future__ import annotations

import pytest

from tools.pv import prepare_lists as pl

HEADER = "stay,period_length,stay_id,y_true,intime,endtime\n"


def _line(subject: int, stay: int) -> str:
    return f"{subject}_episode1_timeseries.csv,48.0,{stay},0,2100-01-01,2100-01-03\n"


@pytest.fixture
def lists(tmp_path):
    """Subjects 1-6 in train; the cohort file lists 1-4 (2 flagged test, as upstream)."""
    folder = tmp_path / "ehr" / "in-hospital-mortality"
    folder.mkdir(parents=True)
    train = [_line(s, 100 + s) for s in range(1, 7)] + [_line(1, 30007216)]
    (folder / "train_listfile.csv").write_text(HEADER + "".join(train), newline="")
    (folder / "val_listfile.csv").write_text(HEADER + _line(7, 107), newline="")
    (folder / "test_listfile.csv").write_text(HEADER + _line(8, 108) + _line(9, 109), newline="")
    testset = tmp_path / "testset_iv.csv"
    testset.write_text("1,0\n2,0\n3,0\n4,1\n")
    return tmp_path / "ehr", folder, testset


def test_filters_drops_the_stay_and_keeps_the_original(lists):
    root, folder, testset = lists
    original = (folder / "train_listfile.csv").read_bytes()
    res = pl.prepare("mortality", root, testset=testset)

    assert (folder / pl.FULL_NAME).read_bytes() == original and res.copied_full
    assert (res.full_rows, res.train_filtered, res.train, res.val, res.test) == (7, 5, 4, 1, 2)
    assert res.dropped_stays == ["30007216"] and res.absent_stays == []
    text = (folder / "train_listfile.csv").read_text()
    assert text.startswith(HEADER) and "30007216" not in text
    assert {ln.split("_")[0] for ln in text.splitlines()[1:]} == {"1", "2", "3", "4"}
    assert set(res.md5) == {
        pl.FULL_NAME,
        "train_listfile.csv",
        "val_listfile.csv",
        "test_listfile.csv",
    }


def test_rerun_is_idempotent_and_never_overwrites_the_original(lists):
    root, folder, testset = lists
    original = (folder / "train_listfile.csv").read_bytes()
    first = pl.prepare("mortality", root, testset=testset)
    built = (folder / "train_listfile.csv").read_bytes()
    second = pl.prepare("mortality", root, testset=testset)

    assert not second.copied_full
    assert (folder / pl.FULL_NAME).read_bytes() == original
    assert (folder / "train_listfile.csv").read_bytes() == built
    assert first.md5 == second.md5


def test_absent_stay_is_reported(lists):
    root, _, testset = lists
    res = pl.prepare("mortality", root, exclude=("424242",), testset=testset)
    assert res.dropped_stays == [] and res.absent_stays == ["424242"] and res.train == 5
    assert "absent from train" in pl.describe(res, "in-hospital-mortality", ("424242",))


def test_stay_rule_asserts_at_most_one_row_per_excluded_stay(lists):
    root, folder, testset = lists
    path = folder / "train_listfile.csv"
    path.write_text(path.read_text() + _line(2, 30007216), newline="")  # duplicate stay id
    with pytest.raises(AssertionError, match="dropped 2 rows"):
        pl.prepare("mortality", root, testset=testset)
    assert not (folder / pl.FULL_NAME).exists()  # nothing written


def test_count_mismatch_is_refused_before_writing(lists):
    root, folder, testset = lists
    original = (folder / "train_listfile.csv").read_bytes()
    with pytest.raises(SystemExit, match="nothing written"):
        pl.prepare(
            "mortality", root, testset=testset, expected=pl.EXPECTED["in-hospital-mortality"]
        )
    assert (folder / "train_listfile.csv").read_bytes() == original
    assert not (folder / pl.FULL_NAME).exists()


def test_cli_prints_counts_and_md5_but_no_rows(lists, capsys):
    root, _, testset = lists
    pl.main(
        [
            "--task",
            "mortality",
            "--listfile-dir",
            str(root),
            "--testset",
            str(testset),
            "--allow-count-mismatch",
        ]
    )
    out = capsys.readouterr().out
    assert "md5 train_listfile.csv" in out and "present in train, dropped" in out
    assert "episode1_timeseries" not in out


def test_expected_counts_are_the_confirmed_ones():
    assert pl.EXPECTED["phenotyping"] == {
        "train_filtered": 42_328,
        "train": 42_327,
        "val": 4_756,
        "test": 11_845,
    }
    assert pl.EXPECTED["in-hospital-mortality"]["train_filtered"] == 19_064
    assert pl.EXPECTED["in-hospital-mortality"]["val"] == 2_161
    assert pl.EXPECTED["in-hospital-mortality"]["test"] == 5_302
    assert pl.DEFAULT_EXCLUDE == ("30007216",)
    assert pl.TESTSET.is_file()  # the bundled code resource (path only; never read here)
