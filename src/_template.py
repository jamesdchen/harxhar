# Vendored prototype of `hpc_agent.template` — the experiment-template +
# parallelization helper layer. See the plan in
# .claude/plans (this-repo-has-the-enchanted-lark.md) and the standalone
# hpc-agent upstream brief. Self-contained and stdlib-only at import time;
# pandas/numpy are imported lazily inside load_series/save_artifact only.
# When `hpc_agent.template` lands upstream this module becomes a re-export.
"""Experiment-template helpers: register experiments, export notebooks, and
declare/plan parallelization.

Layer 1 — helpers
    register_run            decorator: marks an entrypoint, synthesizes the
                            CLI Flag list, injects compute(args)
    save_artifact           large-artifact sink for inside run()
    export_notebook         notebook .ipynb -> .py via a strict AST rule
    discover_runs           AST discovery of @register_run modules
    Flag/flag/generic_args/gpu_args   CLI flag model

Layer 2 — parallelization planner
    Independent/Associative/BoundedHalo/Sequential   data-axis kinds
    plan_tasks              sweep + data-axis -> task list (FLAGS/total/resolve)
    load_series             halo-aware ordered-series loader
    serial_elision_check    fungibility backstop (parallel == serial)
"""

from __future__ import annotations

import ast
import contextlib
import contextvars
import inspect
import itertools
import json
import sys
import types
import typing
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "Flag",
    "flag",
    "generic_args",
    "gpu_args",
    "register_run",
    "RunSpec",
    "save_artifact",
    "export_notebook",
    "discover_runs",
    "Independent",
    "Associative",
    "BoundedHalo",
    "Sequential",
    "plan_tasks",
    "Plan",
    "load_series",
    "serial_elision_check",
]

# ─────────────────────────────────────────────────────────────────────────────
# CLI flag model
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Flag:
    """One CLI flag. Mirrors hpc_agent.executor_cli.Flag."""

    name: str
    type: type = str
    default: Any = None
    required: bool = False
    choices: tuple | None = None
    help: str = ""
    nargs: str | None = None


def flag(name, type=str, default=None, *, required=False, choices=None, help="", nargs=None) -> Flag:
    return Flag(
        name=name,
        type=type,
        default=default,
        required=required,
        choices=tuple(choices) if choices else None,
        help=help,
        nargs=nargs,
    )


def generic_args() -> list[Flag]:
    """Base flags every executor accepts. `start`/`end` are planner-set."""
    return [
        flag("output_file", str, required=True, help="output path for results"),
        flag("data_path", str, default="all30min", help="input dataset path"),
        flag("start", int, default=0, help="chunk start row (planner-set)"),
        flag("end", int, default=-1, help="chunk end row, -1 = end (planner-set)"),
        flag("seed", int, default=42, help="RNG seed"),
    ]


def gpu_args() -> list[Flag]:
    """Extra flags for GPU/DL executors."""
    return [
        flag("gpu_count", int, default=1),
        flag("epochs", int, default=10),
        flag("batch_size", int, default=32),
        flag("learning_rate", float, default=1e-3),
    ]


_GPU_NAMES = frozenset({"gpu_count", "epochs", "batch_size", "learning_rate"})
_GENERIC_NAMES = frozenset(f.name for f in generic_args())

_UNION_ORIGINS: set[Any] = {typing.Union}
if hasattr(types, "UnionType"):  # py3.10+: `X | None`
    _UNION_ORIGINS.add(types.UnionType)


_ANNO_NS = {
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "list": list,
    "dict": dict,
    "tuple": tuple,
    "List": list,
    "Dict": dict,
    "Tuple": typing.Tuple,  # noqa: UP006
    "Optional": typing.Optional,
    "Union": typing.Union,
    "Literal": typing.Literal,
    "Any": typing.Any,
}


def _map_annotation(ann: Any, param_name: str = "") -> tuple[type, tuple | None, str | None]:
    """Map a runtime type annotation -> (Flag.type, choices, nargs)."""
    if ann is inspect.Parameter.empty:
        warnings.warn(f"parameter {param_name!r} has no type annotation; assuming str", stacklevel=3)
        return str, None, None
    if isinstance(ann, str):
        # PEP 563 / `from __future__ import annotations` — resolve the string.
        try:
            ann = eval(ann, {"__builtins__": {}}, _ANNO_NS)  # noqa: S307
        except Exception:  # noqa: BLE001
            warnings.warn(
                f"could not resolve annotation {ann!r} for {param_name!r}; assuming str",
                stacklevel=3,
            )
            return str, None, None
    origin = typing.get_origin(ann)
    args = typing.get_args(ann)
    if origin in _UNION_ORIGINS:
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return _map_annotation(non_none[0], param_name)
        return str, None, None
    if origin is typing.Literal:
        return str, tuple(args), None
    if origin in (list, typing.List):  # noqa: UP006
        inner = args[0] if args else str
        itype, _, _ = _map_annotation(inner, param_name)
        return itype, None, "+"
    if ann in (str, int, float, bool):
        return ann, None, None
    return str, None, None


def _signature_to_flags(sig: inspect.Signature) -> list[Flag]:
    out: list[Flag] = []
    for name, p in sig.parameters.items():
        if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        ftype, choices, nargs = _map_annotation(p.annotation, name)
        required = p.default is inspect.Parameter.empty
        default = None if required else p.default
        out.append(Flag(name, ftype, default, required, choices, "", nargs))
    return out


def _detect_gpu(sig: inspect.Signature) -> bool:
    return any(n in _GPU_NAMES for n in sig.parameters)


def _dedup_by_name(flags: list[Flag]) -> list[Flag]:
    """Dedup by name; a later flag (run's signature) wins over base flags."""
    seen: dict[str, Flag] = {}
    for f in flags:
        seen[f.name] = f
    return list(seen.values())


# ─────────────────────────────────────────────────────────────────────────────
# register_run + compute injection
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RunSpec:
    module: str
    fn_name: str
    flags: tuple[Flag, ...]
    is_gpu: bool


_RUNS: dict[str, RunSpec] = {}

_ARTIFACT_DIR: contextvars.ContextVar = contextvars.ContextVar("hpc_artifact_dir", default=None)
_RANGE: contextvars.ContextVar = contextvars.ContextVar("hpc_range", default=(0, -1))


@contextlib.contextmanager
def _artifact_ctx(output_file):
    token = _ARTIFACT_DIR.set(Path(output_file).parent if output_file else None)
    try:
        yield
    finally:
        _ARTIFACT_DIR.reset(token)


@contextlib.contextmanager
def _range_ctx(start, end):
    token = _RANGE.set((int(start), int(end)))
    try:
        yield
    finally:
        _RANGE.reset(token)


def register_run(fn: Callable | None = None, *, gpu: bool = False):
    """Decorator. Marks an experiment entrypoint. Bare (`@register_run`) or
    factory (`@register_run(gpu=True)`) form.

    Synthesizes the Flag list from the typed signature, builds a `compute(args)`
    wrapper, injects it into the defining module, and records a RunSpec in
    `_RUNS`. The decorator carries no parallelism declaration — parallelism is
    inferred separately (see plan_tasks / the /submit-hpc agent step).
    """
    if fn is None:
        return lambda f: register_run(f, gpu=gpu)

    sig = inspect.signature(fn)
    is_gpu = gpu or _detect_gpu(sig)
    base = list(gpu_args() if is_gpu else []) + list(generic_args())
    all_flags = tuple(_dedup_by_name(base + _signature_to_flags(sig)))

    def compute(args) -> Any:
        kwargs = {p: getattr(args, p) for p in sig.parameters if hasattr(args, p)}
        out = getattr(args, "output_file", None)
        with _artifact_ctx(out), _range_ctx(getattr(args, "start", 0), getattr(args, "end", -1)):
            result = fn(**kwargs)
        if isinstance(result, dict) and out:
            p = Path(out)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(result, default=str, indent=2))
        return result

    mod = inspect.getmodule(fn)
    modname = getattr(mod, "__name__", "__main__")
    if mod is not None:
        # satisfies hpc-agent's `compute(args)` contract
        mod.compute = compute  # type: ignore[attr-defined]
    _RUNS[modname] = RunSpec(modname, fn.__name__, all_flags, is_gpu)

    # Attach for testing / introspection without going through sys.modules.
    fn._hpc_flags = all_flags  # type: ignore[attr-defined]
    fn._hpc_compute = compute  # type: ignore[attr-defined]
    fn._hpc_is_gpu = is_gpu  # type: ignore[attr-defined]
    return fn


def save_artifact(name: str, obj: Any) -> Path:
    """Persist a large artifact next to the current task's output file.

    Resolves the target dir from the `_artifact_ctx` set by `compute`; falls
    back to CWD for in-notebook smoke tests. Serializer chosen by extension.
    """
    base = _ARTIFACT_DIR.get() or Path.cwd()
    base = Path(base)
    base.mkdir(parents=True, exist_ok=True)
    path = base / name
    suffix = path.suffix.lower()
    if suffix == ".parquet" and hasattr(obj, "to_parquet"):
        obj.to_parquet(path)
    elif suffix == ".csv" and hasattr(obj, "to_csv"):
        obj.to_csv(path, index=False)
    elif suffix == ".json":
        path.write_text(json.dumps(obj, default=str, indent=2))
    elif suffix == ".npy":
        import numpy as np

        np.save(path, obj)
    else:
        import pickle

        with open(path, "wb") as fh:
            pickle.dump(obj, fh)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Notebook export — strict AST rule
# ─────────────────────────────────────────────────────────────────────────────

_KEEP_NODES = (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def _is_upper_assign(node: ast.AST) -> bool:
    if isinstance(node, ast.Assign):
        return bool(node.targets) and all(isinstance(t, ast.Name) and t.id.isupper() for t in node.targets)
    if isinstance(node, ast.AnnAssign):
        return isinstance(node.target, ast.Name) and node.target.id.isupper()
    return False


def _decorator_name(dec: ast.AST) -> str | None:
    if isinstance(dec, ast.Call):
        dec = dec.func
    if isinstance(dec, ast.Name):
        return dec.id
    if isinstance(dec, ast.Attribute):
        return dec.attr
    return None


def _register_run_aliases(tree: ast.AST) -> set[str]:
    """Local names that resolve to `register_run` (handles `import as`)."""
    aliases = {"register_run"}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for a in node.names:
                if a.name == "register_run":
                    aliases.add(a.asname or a.name)
    return aliases


def _is_register_run(node: ast.AST, aliases: set[str]) -> bool:
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return False
    return any(_decorator_name(d) in aliases for d in node.decorator_list)


def _node_source(src: str, node: ast.stmt) -> str | None:
    """Source for a top-level node, INCLUDING decorators.

    ast.get_source_segment omits decorator lines because a decorated
    FunctionDef/ClassDef reports `lineno` at the `def`/`class` keyword; we
    extend the span back to the first decorator.
    """
    if node.end_lineno is None:
        return None
    start = node.lineno
    deco = getattr(node, "decorator_list", None)
    if deco:
        start = min(start, min(d.lineno for d in deco))
    return "\n".join(src.splitlines()[start - 1 : node.end_lineno])


def export_notebook(nb_path: str | Path, out_path: str | Path) -> Path:
    """Export a notebook to a .py module by a strict structural AST rule.

    Emits, in source order, only: imports, function/class defs, and
    UPPERCASE-target assignments. Everything else (smoke tests, plots, lowercase
    assignments, bare expressions) is skipped silently. `from __future__`
    imports are hoisted to the top and de-duplicated.

    Raises ValueError if the notebook has no @register_run-decorated function.
    """
    nb = json.loads(Path(nb_path).read_text())
    cells = [
        ("".join(c["source"]) if isinstance(c["source"], list) else c["source"])
        for c in nb.get("cells", [])
        if c.get("cell_type") == "code"
    ]
    parsed: list[tuple[str, ast.Module]] = []
    aliases: set[str] = {"register_run"}
    for src in cells:
        if not src.strip():
            continue
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue  # cell with IPython magics or an incomplete fragment
        parsed.append((src, tree))
        aliases |= _register_run_aliases(tree)

    future_lines: list[str] = []
    kept: list[str] = []
    has_register_run = False
    for src, tree in parsed:
        for node in tree.body:
            seg = _node_source(src, node)
            if seg is None:
                continue
            if isinstance(node, ast.ImportFrom) and node.module == "__future__":
                for line in seg.splitlines():
                    if line.strip() and line not in future_lines:
                        future_lines.append(line)
                continue
            if isinstance(node, _KEEP_NODES) or _is_upper_assign(node):
                kept.append(seg)
                if _is_register_run(node, aliases):
                    has_register_run = True

    if not has_register_run:
        raise ValueError(f"{nb_path}: no @register_run-decorated function found")

    header = f"# Auto-generated from {Path(nb_path).name}. Do not edit by hand."
    blocks = [header]
    if future_lines:
        blocks.append("\n".join(future_lines))
    blocks.append("\n\n".join(kept))
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n\n".join(blocks) + "\n")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# AST discovery — for the framework env (no heavy imports)
# ─────────────────────────────────────────────────────────────────────────────

_AST_TYPE_NAMES = {"str": str, "int": int, "float": float, "bool": bool}


def _ast_anno_to_flag_bits(anno: ast.AST | None) -> tuple[type, tuple | None, str | None]:
    """(type, choices, nargs) from an AST annotation node."""
    if anno is None:
        return str, None, None
    if isinstance(anno, ast.Name):
        return _AST_TYPE_NAMES.get(anno.id, str), None, None
    if isinstance(anno, ast.Constant) and anno.value is None:
        return str, None, None
    if isinstance(anno, ast.BinOp) and isinstance(anno.op, ast.BitOr):  # X | None
        for side in (anno.left, anno.right):
            if isinstance(side, ast.Constant) and side.value is None:
                other = anno.right if side is anno.left else anno.left
                return _ast_anno_to_flag_bits(other)
        return str, None, None
    if isinstance(anno, ast.Subscript):
        base = anno.value.id if isinstance(anno.value, ast.Name) else ""
        if base in ("list", "List"):
            inner = anno.slice
            itype, _, _ = _ast_anno_to_flag_bits(inner)
            return itype, None, "+"
        if base == "Optional":
            return _ast_anno_to_flag_bits(anno.slice)
        if base == "Literal":
            elts = anno.slice.elts if isinstance(anno.slice, ast.Tuple) else [anno.slice]
            choices = tuple(e.value for e in elts if isinstance(e, ast.Constant))
            return str, choices, None
    return str, None, None


def _ast_function_to_flags(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[Flag]:
    posargs = list(node.args.args)
    defaults = list(node.args.defaults)
    n_required = len(posargs) - len(defaults)
    flags: list[Flag] = []
    for i, arg in enumerate(posargs):
        if arg.arg == "self":
            continue
        ftype, choices, nargs = _ast_anno_to_flag_bits(arg.annotation)
        if i < n_required:
            flags.append(Flag(arg.arg, ftype, None, True, choices, "", nargs))
        else:
            dnode = defaults[i - n_required]
            try:
                default = ast.literal_eval(dnode)
            except (ValueError, SyntaxError):
                default = None
            flags.append(Flag(arg.arg, ftype, default, False, choices, "", nargs))
    return flags


def discover_runs(src_dir: str | Path) -> dict[str, RunSpec]:
    """AST-walk `src_dir/*.py` for @register_run functions. No heavy imports —
    safe in the stdlib-only framework env."""
    src_dir = Path(src_dir)
    out: dict[str, RunSpec] = {}
    for py in sorted(src_dir.glob("*.py")):
        if py.name.startswith("_"):
            continue
        try:
            tree = ast.parse(py.read_text())
        except SyntaxError:
            continue
        aliases = _register_run_aliases(tree)
        for node in tree.body:
            if not _is_register_run(node, aliases):
                continue
            assert isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            modname = f"{src_dir.name}.{py.stem}"
            sig_flags = _ast_function_to_flags(node)
            names = {a.arg for a in node.args.args}
            is_gpu = bool(_GPU_NAMES & names)
            base = list(gpu_args() if is_gpu else []) + list(generic_args())
            flags = tuple(_dedup_by_name(base + sig_flags))
            out[modname] = RunSpec(modname, node.name, flags, is_gpu)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Parallelization planner
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Independent:
    """No carried state across the data axis — split anywhere."""


@dataclass(frozen=True)
class Sequential:
    """Unbounded / order-dependent state — do not split this axis."""


@dataclass(frozen=True)
class Associative:
    """Carried state with an associative combine — parallel prefix-scan."""

    combine: Callable | None = None


@dataclass(frozen=True)
class BoundedHalo:
    """Carried state of bounded backward distance — halo overlap.

    `halo` maps a params object (attribute access on the sweep point) to the
    look-back width in rows.
    """

    halo: Callable[[Any], int]


DataAxis = Independent | Sequential | Associative | BoundedHalo


class _Params(dict):
    """dict with attribute access, so halo=lambda p: p.train_window works."""

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as e:
            raise AttributeError(k) from e


def _range_split(total_rows: int, total_chunks: int, chunk_id: int, overlap: int) -> tuple[int, int, int]:
    """Split [overlap, total_rows) into total_chunks; prepend `overlap` look-back.

    Returns (start, end, emit_start): the executor reads [start, end) and emits
    predictions only for [emit_start, end). Chunks tile [overlap, total_rows)
    exactly. Mirrors harxhar's range_split_overlap.
    """
    oos_rows = total_rows - overlap
    base, rem = divmod(oos_rows, total_chunks)
    oos_start = overlap + base * chunk_id + min(chunk_id, rem)
    oos_end = oos_start + base + (1 if chunk_id < rem else 0)
    return oos_start - overlap, oos_end, oos_start


@dataclass
class Plan:
    """A materialized task list. `tasks.py` exposes total()/resolve() from this."""

    tasks: list[dict]

    def total(self) -> int:
        return len(self.tasks)

    def resolve(self, task_id: int) -> dict:
        return dict(self.tasks[task_id])


def plan_tasks(
    sweep: dict[str, list] | None = None,
    data_axis: DataAxis | None = None,
    chunks: int = 1,
    total_rows: int | None = None,
) -> Plan:
    """Build the task list from a sweep grid + a data-axis classification.

    - Independent      : split [0, total_rows) into `chunks` (no halo)
    - Associative      : split with no halo (reduce carries the summary)
    - BoundedHalo      : split with a per-task look-back halo
    - Sequential       : never split — one task per sweep point
    """
    sweep = sweep or {}
    data_axis = data_axis or Sequential()
    if sweep:
        names = list(sweep)
        points = [dict(zip(names, combo, strict=True)) for combo in itertools.product(*(sweep[n] for n in names))]
    else:
        points = [{}]

    tasks: list[dict] = []
    for pt in points:
        if isinstance(data_axis, Sequential) or chunks <= 1:
            tasks.append({**pt, "start": 0, "end": -1})
            continue
        if total_rows is None:
            raise ValueError("plan_tasks: total_rows is required to split a data axis")
        overlap = 0
        if isinstance(data_axis, BoundedHalo):
            overlap = int(data_axis.halo(_Params(pt)))
        for c in range(chunks):
            start, end, emit_start = _range_split(total_rows, chunks, c, overlap)
            task = {**pt, "start": start, "end": end}
            if overlap:
                task["emit_start"] = emit_start
            tasks.append(task)
    return Plan(tasks)


def load_series(name: str, data_dir: str | Path = "data"):
    """Load an ordered series. On a chunked task (a `_range_ctx` is active) it
    returns only that task's [start, end) slice; otherwise the whole series.

    This is the one seam that lets the framework control the slice without the
    experiment knowing. Imports pandas lazily.
    """
    import pandas as pd

    path = Path(name)
    if not path.exists():
        path = Path(data_dir) / name
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix in (".csv", ".txt"):
        df = pd.read_csv(path)
    else:
        raise ValueError(f"load_series: unsupported file type {path.suffix!r}")

    start, end = _RANGE.get()
    if (start, end) != (0, -1):
        df = df.iloc[start : (None if end == -1 else end)].reset_index(drop=True)
    return df


def serial_elision_check(
    scan_fn: Callable[[list], list],
    inputs: list,
    *,
    chunks: int,
    halo: int,
    tol: float = 1e-9,
) -> bool:
    """Fungibility backstop: assert a chunked run equals the serial run.

    `scan_fn(seq)` must return a per-row output sequence aligned to `seq`.
    Runs `scan_fn` once over the whole input and once per `_range_split` chunk
    (with `halo` look-back), trims each chunk to its emitted region, and
    asserts the concatenation matches the serial output over [halo, len).

    Returns True; raises AssertionError on a mismatch (a wrong axis/halo).
    """
    whole = list(scan_fn(inputs))
    n = len(inputs)
    pieces: list = []
    for c in range(chunks):
        start, end, emit_start = _range_split(n, chunks, c, halo)
        chunk_out = list(scan_fn(inputs[start:end]))
        pieces.extend(chunk_out[emit_start - start :])

    expected = whole[halo:]
    if len(pieces) != len(expected):
        raise AssertionError(f"serial-elision: chunked produced {len(pieces)} rows, serial {len(expected)}")
    for i, (a, b) in enumerate(zip(expected, pieces, strict=True)):
        if abs(a - b) > tol:
            raise AssertionError(f"serial-elision mismatch at emitted row {i}: {a!r} != {b!r}")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# CLI: notebook export entrypoint (used by `make export` / pre-commit)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    argv = sys.argv[1:]
    if argv and argv[0] == "export":
        argv = argv[1:]
    if len(argv) != 2:
        print("usage: python -m src._template export <notebook.ipynb> <out.py>", file=sys.stderr)
        sys.exit(2)
    dest = export_notebook(argv[0], argv[1])
    print(f"exported {argv[0]} -> {dest}")
