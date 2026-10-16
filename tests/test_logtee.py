"""tools/pv/logtee.py and the --log option of smoke_round3 / r3_preflight."""

from __future__ import annotations

import sys

import pytest

from tools.pv import logtee, r3_preflight, smoke_round3


def test_tee_keeps_prints_subprocess_output_and_the_traceback(tmp_path, capsys):
    path = tmp_path / "logs" / "x.log"
    with pytest.raises(RuntimeError), logtee.tee_to_log(path):
        print("hello from the tool")
        code = logtee.run_logged([sys.executable, "-c", "print('from a subprocess')"])
        assert code == 0
        raise RuntimeError("the real error")
    text = path.read_text(encoding="utf-8")
    for expected in ("hello from the tool", "from a subprocess", "RuntimeError: the real error",
                     f"full output also in {path}"):  # fmt: skip
        assert expected in text, expected
    assert f"[log] full output in {path}" in capsys.readouterr().out  # printed at the end


@pytest.mark.parametrize("module", [smoke_round3, r3_preflight])
def test_main_strips_log_and_writes_it(module, tmp_path, monkeypatch):
    seen = []

    def fake_run(argv):
        seen.append(argv)
        print("step output")
        return 0

    monkeypatch.setattr(module, "_run", fake_run)
    path = tmp_path / "out.log"
    assert module.main(["--task", "mortality", "--log", str(path)]) == 0
    assert seen == [["--task", "mortality"]]
    assert "step output" in path.read_text(encoding="utf-8")


def test_default_log_is_under_runs_root(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNS_ROOT", str(tmp_path))
    path, rest = smoke_round3._log_path(["--task", "mortality"], "smoke_round3")
    assert path.parent == tmp_path / "logs" and path.name.startswith("smoke_round3_mortality_")
    assert rest == ["--task", "mortality"]


def test_tokenizers_parallelism_is_off_for_tools_and_fusion_main():
    import os  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    import tools.pv.medpatch_bridge  # noqa: F401,PLC0415

    assert os.environ["TOKENIZERS_PARALLELISM"] == "false"
    main = (Path(__file__).resolve().parents[1] / "medpatch" / "fusion_main.py").read_text()
    head = main.split("from trainers", 1)[0]  # set before anything imports transformers
    assert 'os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")' in head
