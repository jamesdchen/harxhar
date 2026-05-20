"""Unit tests for src/_template.py — the vendored experiment-template helpers."""

from __future__ import annotations

import argparse
import json
import textwrap

import pytest

from src import _template as t

# ── Flag synthesis ───────────────────────────────────────────────────────────


def test_signature_to_flags_type_mapping():
    import inspect
    from typing import Literal

    def fn(
        req: str,
        n: int = 1,
        x: float = 0.5,
        flag_: bool = False,
        maybe: str | None = None,
        pipe: str | None = None,
        choice: Literal["a", "b"] = "a",
        many: list[int] = None,  # noqa: RUF013
    ): ...

    flags = {f.name: f for f in t._signature_to_flags(inspect.signature(fn))}
    assert flags["req"].type is str and flags["req"].required
    assert flags["n"].type is int and flags["n"].default == 1
    assert flags["x"].type is float
    assert flags["flag_"].type is bool
    assert flags["maybe"].type is str and not flags["maybe"].required
    assert flags["pipe"].type is str  # X | None unwrapped
    assert flags["choice"].choices == ("a", "b")
    assert flags["many"].type is int and flags["many"].nargs == "+"


def test_detect_gpu():
    import inspect

    def cpu(horizon: int = 1): ...
    def gpu(epochs: int = 10, batch_size: int = 32): ...

    assert not t._detect_gpu(inspect.signature(cpu))
    assert t._detect_gpu(inspect.signature(gpu))


# ── register_run + compute ───────────────────────────────────────────────────


def test_register_run_synthesizes_flags_and_compute(tmp_path):
    @t.register_run
    def run(horizon: int = 1, seed: int = 42) -> dict:
        return {"horizon": horizon, "seed": seed}

    names = {f.name for f in run._hpc_flags}
    # generic flags + signature params, deduped
    assert {"output_file", "data_path", "start", "end", "seed", "horizon"} <= names
    # `seed` appears once (signature wins over generic_args)
    assert sum(f.name == "seed" for f in run._hpc_flags) == 1

    out = tmp_path / "r.json"
    run._hpc_compute(argparse.Namespace(horizon=3, seed=7, output_file=str(out), start=0, end=-1))
    assert json.loads(out.read_text()) == {"horizon": 3, "seed": 7}


def test_register_run_factory_form_gpu():
    @t.register_run(gpu=True)
    def run(horizon: int = 1) -> dict:
        return {}

    assert run._hpc_is_gpu
    assert "gpu_count" in {f.name for f in run._hpc_flags}


def test_save_artifact_json_falls_back_to_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = t.save_artifact("m.json", {"a": 1})
    assert path.exists() and json.loads(path.read_text()) == {"a": 1}


# ── export_notebook (strict AST rule) ────────────────────────────────────────


def _nb(*cell_sources: str) -> dict:
    return {
        "cells": [
            {"cell_type": "code", "source": src, "metadata": {}, "outputs": [], "execution_count": None}
            for src in cell_sources
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def test_export_notebook_keeps_structure_drops_scratch(tmp_path):
    nb = _nb(
        "from __future__ import annotations\nimport numpy as np\n",
        "MAX_LAG = 22\n",
        "def helper(x):\n    return x + 1\n",
        "@register_run\ndef run(seed: int = 0) -> dict:\n    return {'s': seed}\n",
        "df = helper(3)\nprint(df)\n",  # scratch — dropped
        "run(seed=1)\n",  # smoke test — dropped
    )
    nb_path = tmp_path / "exp.ipynb"
    nb_path.write_text(json.dumps(nb))
    out = tmp_path / "exp.py"
    t.export_notebook(nb_path, out)

    text = out.read_text()
    assert "import numpy as np" in text
    assert "MAX_LAG = 22" in text
    assert "def helper(x):" in text
    assert "@register_run" in text  # decorator must survive export
    assert "def run(" in text
    assert "df = helper(3)" not in text  # lowercase assign dropped
    assert "print(df)" not in text
    assert "run(seed=1)" not in text  # bare call dropped
    assert text.index("from __future__") < text.index("import numpy")  # future hoisted


def test_export_notebook_requires_register_run(tmp_path):
    nb_path = tmp_path / "no_run.ipynb"
    nb_path.write_text(json.dumps(_nb("def helper():\n    return 1\n")))
    with pytest.raises(ValueError, match="register_run"):
        t.export_notebook(nb_path, tmp_path / "out.py")


def test_export_notebook_skips_magic_cells(tmp_path):
    nb = _nb(
        "%matplotlib inline\n",  # magic — unparseable, skipped
        "@register_run\ndef run() -> dict:\n    return {}\n",
    )
    nb_path = tmp_path / "m.ipynb"
    nb_path.write_text(json.dumps(nb))
    out = t.export_notebook(nb_path, tmp_path / "m.py")
    assert "def run()" in out.read_text()


# ── discover_runs (AST, no import) ───────────────────────────────────────────


def test_discover_runs_finds_decorated_functions(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "exp_a.py").write_text(
        textwrap.dedent("""
        from src._template import register_run

        @register_run
        def run(horizon: int = 1, tag: str = "x"):
            return {}
    """)
    )
    (src / "_helper.py").write_text("X = 1\n")  # underscore — skipped

    found = t.discover_runs(src)
    assert set(found) == {"src.exp_a"}
    flags = {f.name: f for f in found["src.exp_a"].flags}
    assert flags["horizon"].type is int and flags["horizon"].default == 1
    assert flags["tag"].type is str
    assert "output_file" in flags  # generic flags merged


def test_discover_runs_resolves_aliased_decorator(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "exp_b.py").write_text(
        textwrap.dedent("""
        from src._template import register_run as rr

        @rr
        def run(n: int = 2):
            return {}
    """)
    )
    found = t.discover_runs(src)
    assert "src.exp_b" in found


# ── planner: plan_tasks + halo math ──────────────────────────────────────────


def test_plan_tasks_sequential_one_task_per_sweep_point():
    plan = t.plan_tasks(sweep={"horizon": [1, 5], "win": [500, 1000]}, data_axis=t.Sequential())
    assert plan.total() == 4
    assert all(task["start"] == 0 and task["end"] == -1 for i in range(4) for task in [plan.resolve(i)])


def test_plan_tasks_bounded_halo_tiles_exactly():
    # halo derived per-task from the sweep point
    plan = t.plan_tasks(
        sweep={"train_window": [10]},
        data_axis=t.BoundedHalo(halo=lambda p: p.train_window),
        chunks=5,
        total_rows=110,
    )
    assert plan.total() == 5
    tasks = [plan.resolve(i) for i in range(5)]
    # OOS regions [emit_start, end) must tile [halo, total_rows) with no gap
    assert tasks[0]["emit_start"] == 10
    for a, b in zip(tasks, tasks[1:], strict=False):
        assert a["end"] == b["emit_start"]
    assert tasks[-1]["end"] == 110
    # each chunk carries exactly `halo` look-back
    for task in tasks:
        assert task["emit_start"] - task["start"] == 10


def test_plan_tasks_bounded_halo_requires_total_rows():
    with pytest.raises(ValueError, match="total_rows"):
        t.plan_tasks(data_axis=t.BoundedHalo(halo=lambda p: 1), chunks=4)


# ── serial elision backstop ──────────────────────────────────────────────────


def test_serial_elision_passes_for_correct_halo():
    # a bounded-halo scan: out[i] = sum(inputs[i-3 .. i])
    win = 3

    def scan(seq):
        return [sum(seq[max(0, i - win) : i + 1]) for i in range(len(seq))]

    data = list(range(1, 41))
    assert t.serial_elision_check(scan, data, chunks=4, halo=win)


def test_serial_elision_fails_for_too_small_halo():
    win = 3

    def scan(seq):
        return [sum(seq[max(0, i - win) : i + 1]) for i in range(len(seq))]

    data = list(range(1, 41))
    with pytest.raises(AssertionError, match="serial-elision"):
        t.serial_elision_check(scan, data, chunks=4, halo=win - 2)  # halo too small
