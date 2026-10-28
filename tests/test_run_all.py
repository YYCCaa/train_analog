"""Parallel suite runner (tests/run_all.py + its pytest plugin tests/run_all_shard.py), driven the way a person uses it:
from the command line, on throw-away project roots full of fake test files, judging only what is printed, the exit
code, the log files and the marks the fake tests leave behind.

    make_root(tmp_path, {...})              run_runner(--root R, --log-dir L, ...)        what the user checks
    R/ pytest.ini                    ──►    python tests/run_all.py                ──►    exit code (0 / 1 / 2)
    R/ tests/test_*.py                         │ one `python -m pytest` per unit           last line "... -> PASS|FAIL"
    R/ pkg/tests/test_*.py                     ▼ (shard i/K: every K-th test)               failure section + log tail
                                         L/ <unit>.log  (one per unit)                     marks written by fake tests

The plugin itself is also driven directly (`python -m pytest -p run_all_shard --shard i/K`), as a user would when
re-running one shard by hand.
"""

from __future__ import annotations

import importlib.util
import os
import re
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

# REPO: Path, repo root (the folder holding tests/run_all.py)
REPO = Path(__file__).resolve().parents[1]
# RUNNER: Path, the runner script under test
RUNNER = REPO / "tests" / "run_all.py"
# PLUGIN_DIR: Path, folder the runner puts on the children's PYTHONPATH so `-p PLUGIN` imports from any --root
PLUGIN_DIR = REPO / "tests"
# PLUGIN: str, importable module name of the shard plugin
PLUGIN = "run_all_shard"
# THREAD_VARS: tuple[str, ...], thread-cap environment variables the runner sets for its children
THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
# REAL_TEST_FILES: list[str], the 21 tracked test files of the repo (repo-relative, posix) that --list must show
REAL_TEST_FILES = [
    "grayscale_webapp/tests/test_api.py", "grayscale_webapp/tests/test_converter.py",
    "grayscale_webapp/tests/test_e2e_browser.py",
    "tests/test_autocal_dataset.py", "tests/test_autocal_math.py", "tests/test_autocal_models.py",
    "tests/test_export_v9_tflite.py", "tests/test_studio_api.py", "tests/test_studio_e2e.py",
    "tests/test_train_gray.py", "tests/test_train_v9.py", "tests/test_v4_generator.py", "tests/test_v7_config.py",
    "tests/test_v7_gray_generator.py", "tests/test_v7_look.py", "tests/test_v7_render.py",
    "tip_webapp/tests/test_api.py", "tip_webapp/tests/test_e2e_browser.py", "tip_webapp/tests/test_inference.py",
    "val_viewer/tests/test_api.py", "val_viewer/tests/test_e2e_browser.py",
]

# PASSING_A: str, fake test file with 2 passing tests
PASSING_A = """
    def test_one():
        assert True

    def test_two():
        assert 1 + 1 == 2
"""
# PASSING_B: str, fake test file with 1 passing test
PASSING_B = """
    def test_three():
        pass
"""


# ── helpers ──────────────────────────────────────────────────────────────────────────────────────────────────────

def make_root(tmp_path: Path, files: dict[str, str], name: str = "root") -> Path:
    """Write a fake project root under tmp_path (Path): a pytest.ini plus `files` (dict[str, str], repo-relative posix
    path -> source, dedented). Returns the root (Path)."""
    root = tmp_path / name
    root.mkdir()
    (root / "pytest.ini").write_text("[pytest]\n")
    for rel, src in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(src))
    return root


def run_runner(*args: str, env: dict[str, str] | None = None, timeout: float = 240,
               normal_priority: bool = False) -> subprocess.CompletedProcess:
    """Run `python tests/run_all.py *args` like a user would; returns the CompletedProcess (stdout: str, stderr
    merged into stdout, decoded with the locale encoding the runner prints in). `normal_priority` (bool): on Windows
    start the runner at normal CPU priority, as from a plain console, even when this test itself runs below normal
    (e.g. as a unit of the real suite)."""
    extra = {"creationflags": subprocess.NORMAL_PRIORITY_CLASS} if normal_priority and os.name == "nt" else {}
    return subprocess.run([sys.executable, str(RUNNER), *args], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, errors="replace", timeout=timeout, env=env, cwd=str(REPO), **extra)


def last_line(out: str) -> str:
    """Last non-empty line (str) of the runner's output `out` (str)."""
    lines = [ln for ln in out.splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def pid_alive(pid: int) -> bool:
    """True when a process with id `pid` (int) still exists (bool)."""
    if os.name == "nt":
        res = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                             capture_output=True, text=True, errors="replace", timeout=30)
        return f'"{pid}"' in res.stdout
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def load_runner_module():
    """Import tests/run_all.py by path (module object), to read its SPLIT table without running anything."""
    spec = importlib.util.spec_from_file_location("run_all_under_test", RUNNER)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclasses resolve their (postponed) annotations through sys.modules
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.modules.pop(spec.name, None)
    return mod


def plugin_env() -> dict[str, str]:
    """Environment (dict[str, str]) in which `-p run_all_shard` imports: PLUGIN_DIR prepended to PYTHONPATH."""
    env = dict(os.environ)
    old = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(PLUGIN_DIR) + (os.pathsep + old if old else "")
    return env


# ── 1. two files, all passing ────────────────────────────────────────────────────────────────────────────────────

def test_runs_every_file_and_passes(tmp_path):
    """User runs the runner on a root with tests/test_a.py (2 tests) and pkg/tests/test_b.py (1 test) -> exit 0,
    last line says 3 passed and ends in PASS, one log file per unit in --log-dir."""
    root = make_root(tmp_path, {"tests/test_a.py": PASSING_A, "pkg/tests/test_b.py": PASSING_B})
    log_dir = tmp_path / "logs"
    res = run_runner("--root", str(root), "--log-dir", str(log_dir))
    assert res.returncode == 0, res.stdout
    final = last_line(res.stdout)
    assert re.search(r"\b3 passed\b", final), final
    assert final.endswith("PASS"), final
    logs = sorted(p.name for p in log_dir.glob("*.log"))  # list[str], one log per unit
    assert len(logs) == 2, logs
    assert any("test_a" in n for n in logs) and any("test_b" in n for n in logs), logs
    for p in log_dir.glob("*.log"):
        assert "passed" in p.read_text(errors="replace"), p


# ── 2. a failing test ────────────────────────────────────────────────────────────────────────────────────────────

def test_failure_is_reported_with_log_tail(tmp_path):
    """User runs it on a root where one test fails with `assert 1 == 2, "boom-marker"` -> exit 1, last line ends in
    FAIL, and the failure section names that unit and shows boom-marker from its log."""
    root = make_root(tmp_path, {
        "tests/test_good.py": PASSING_B,
        "tests/test_bad.py": """
            def test_bad():
                assert 1 == 2, "boom-marker"
        """,
    })
    log_dir = tmp_path / "logs"
    res = run_runner("--root", str(root), "--log-dir", str(log_dir))
    assert res.returncode == 1, res.stdout
    assert last_line(res.stdout).endswith("FAIL"), res.stdout
    lines = res.stdout.splitlines()
    # header: int, index of the failure-section line naming the bad unit and its log path
    header = next(i for i, ln in enumerate(lines)
                  if "tests__test_bad.py" in ln and str(log_dir / "tests__test_bad.py.log") in ln)
    assert any("boom-marker" in ln for ln in lines[header + 1:]), res.stdout
    assert not any("tests__test_good.py" in ln and ".log" in ln for ln in lines), "passing unit's log was dumped"


# ── 3. timeout kills the whole process tree ──────────────────────────────────────────────────────────────────────

def test_timeout_kills_process_tree(tmp_path):
    """User runs it with --timeout 5 on a test that sleeps 120 s -> returns within ~30 s, exit 1, the unit is
    reported as TIMEOUT, and the sleeping test's (real interpreter) process is gone."""
    pid_file = tmp_path / "sleeper.pid"
    root = make_root(tmp_path, {"tests/test_sleep.py": f"""
        import os, pathlib, time

        def test_sleeps():
            pathlib.Path({str(pid_file)!r}).write_text(str(os.getpid()))
            time.sleep(120)
    """})
    t0 = time.monotonic()
    res = run_runner("--root", str(root), "--log-dir", str(tmp_path / "logs"), "--timeout", "5", timeout=120)
    elapsed = time.monotonic() - t0  # float, seconds the runner took
    assert elapsed < 30, (elapsed, res.stdout)
    assert res.returncode == 1, res.stdout
    assert any("TIMEOUT" in ln and "tests/test_sleep.py" in ln for ln in res.stdout.splitlines()), res.stdout
    assert last_line(res.stdout).endswith("FAIL"), res.stdout
    pid = int(pid_file.read_text())  # int, pid of the interpreter that ran the sleeping test
    deadline = time.monotonic() + 15
    while pid_alive(pid) and time.monotonic() < deadline:
        time.sleep(0.5)
    assert not pid_alive(pid), f"sleeping test process {pid} survived the timeout"


# ── 4. a split file runs every test exactly once ─────────────────────────────────────────────────────────────────

def test_split_runs_each_test_once(tmp_path):
    """User runs it with --split tests/test_c.py=3 on a file with 7 tests -> 3 units, every test ran exactly once
    (round-robin: shard i got tests i, i+3, ...), exit 0, total says 7 passed."""
    marks = tmp_path / "marks"
    marks.mkdir()
    body = "".join(f"\ndef test_c{n}():\n    _mark('test_c{n}')\n" for n in range(7))
    root = make_root(tmp_path, {"tests/test_c.py": f"""
import os, pathlib

def _mark(name):
    with open(pathlib.Path({str(marks)!r}) / f"{{os.getpid()}}.txt", "a") as f:
        f.write(name + "\\n")
{body}"""})
    log_dir = tmp_path / "logs"
    res = run_runner("--root", str(root), "--log-dir", str(log_dir), "--split", "tests/test_c.py=3")
    assert res.returncode == 0, res.stdout
    final = last_line(res.stdout)
    assert re.search(r"\b7 passed\b", final) and final.endswith("PASS"), final
    assert len(list(log_dir.glob("*.log"))) == 3, sorted(p.name for p in log_dir.glob("*.log"))
    for i in (1, 2, 3):
        assert f"tests/test_c.py [{i}/3]" in res.stdout, res.stdout
    per_proc = [p.read_text().split() for p in marks.glob("*.txt")]  # list[list[str]], test names per process
    ran = sorted(n for names in per_proc for n in names)  # list[str], every test that ran
    assert ran == [f"test_c{n}" for n in range(7)], ran
    assert sorted(sorted(names) for names in per_proc) == [
        ["test_c0", "test_c3", "test_c6"], ["test_c1", "test_c4"], ["test_c2", "test_c5"]], per_proc


# ── 5. empty shards are fine, a file without tests is not ────────────────────────────────────────────────────────

def test_empty_shards_pass(tmp_path):
    """User runs it with --split tests/test_d.py=5 on a file with 2 tests -> exit 0 (3 shards are empty)."""
    root = make_root(tmp_path, {"tests/test_d.py": PASSING_A})
    res = run_runner("--root", str(root), "--log-dir", str(tmp_path / "logs"), "--split", "tests/test_d.py=5")
    assert res.returncode == 0, res.stdout
    final = last_line(res.stdout)
    assert re.search(r"\b2 passed\b", final) and final.endswith("PASS"), final


@pytest.mark.parametrize("split", [None, "tests/test_e.py=2"], ids=["whole", "split"])
def test_file_without_tests_fails(tmp_path, split):
    """User runs it on a root with a file holding 0 tests (only a helper) -> exit 1 with "no tests" for that file,
    whether the file is run whole or split."""
    root = make_root(tmp_path, {
        "tests/test_ok.py": PASSING_B,
        "tests/test_e.py": """
            def helper():
                return 1
        """,
    })
    extra = ["--split", split] if split else []  # list[str], optional --split argument
    res = run_runner("--root", str(root), "--log-dir", str(tmp_path / "logs"), *extra)
    assert res.returncode == 1, res.stdout
    assert any("no tests" in ln and "tests/test_e.py" in ln for ln in res.stdout.splitlines()), res.stdout
    assert last_line(res.stdout).endswith("FAIL"), res.stdout


# ── 6. positional files restrict the run ─────────────────────────────────────────────────────────────────────────

def test_positional_file_restricts_run(tmp_path):
    """User runs it with tests/test_a.py as positional argument on the root of case 1 -> only that file runs."""
    root = make_root(tmp_path, {"tests/test_a.py": PASSING_A, "pkg/tests/test_b.py": PASSING_B})
    log_dir = tmp_path / "logs"
    res = run_runner("--root", str(root), "--log-dir", str(log_dir), "tests/test_a.py")
    assert res.returncode == 0, res.stdout
    final = last_line(res.stdout)
    assert re.search(r"\b2 passed\b", final) and final.endswith("PASS"), final
    assert "test_b" not in res.stdout, res.stdout
    assert [p.name for p in log_dir.glob("*.log")] == ["tests__test_a.py.log"]


# ── 7. --list on the real repo ───────────────────────────────────────────────────────────────────────────────────

def test_list_real_repo(tmp_path):
    """User runs --list on the real repo root (no --root) -> exit 0 within 20 s, every tracked test file listed
    (each shard of a split file as `file [i/K]`), nothing run and no log folder created."""
    split = load_runner_module().SPLIT  # dict[str, int], the runner's built-in split table
    log_dir = tmp_path / "never_created"
    t0 = time.monotonic()
    res = run_runner("--list", "--log-dir", str(log_dir), timeout=60)
    elapsed = time.monotonic() - t0  # float, seconds --list took
    assert res.returncode == 0, res.stdout
    assert elapsed < 20, elapsed
    for rel in REAL_TEST_FILES + ["tests/test_run_all.py"]:
        k = split.get(rel, 1)  # int, number of units this file must show
        if k == 1:
            assert re.search(rf"(?m)\s{re.escape(rel)}\s*(\(|$)", res.stdout), (rel, res.stdout)
        else:
            for i in range(1, k + 1):
                assert f"{rel} [{i}/{k}]" in res.stdout, (rel, i, res.stdout)
    assert "tests/test_studio_e2e.py [" not in res.stdout, "studio e2e must never be split"
    assert " passed" not in res.stdout, res.stdout
    assert not log_dir.exists(), "--list must not create the log folder"


# ── 8. thread caps and the plugin's argument check ───────────────────────────────────────────────────────────────

# ENV_TEST: str, fake test comparing every thread variable with EXPECT_<var> from the environment
ENV_TEST = """
    import os

    VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")

    def test_thread_caps():
        for v in VARS:
            assert os.environ.get(v) == os.environ["EXPECT_" + v], v
"""


@pytest.mark.parametrize("preset", [{}, {"OMP_NUM_THREADS": "7"}], ids=["unset", "omp-preset"])
def test_threads_option_sets_caps(tmp_path, preset):
    """User runs it with --threads 3 -> a fake test sees OMP_NUM_THREADS == "3" (and the other 3 caps); a variable
    the user already set in the runner's environment is kept."""
    root = make_root(tmp_path, {"tests/test_env.py": ENV_TEST})
    env = {k: v for k, v in os.environ.items() if k not in THREAD_VARS}  # dict[str, str], runner environment
    env.update(preset)
    for v in THREAD_VARS:
        env["EXPECT_" + v] = preset.get(v, "3")
    res = run_runner("--root", str(root), "--log-dir", str(tmp_path / "logs"), "--threads", "3", env=env)
    assert res.returncode == 0, res.stdout
    assert re.search(r"\b1 passed\b", last_line(res.stdout)), res.stdout


def run_plugin(root: Path, *args: str) -> subprocess.CompletedProcess:
    """Run `python -m pytest -q -p no:cacheprovider -p run_all_shard *args` in `root` (Path) by hand; returns the
    CompletedProcess (stdout: str, stderr merged)."""
    return subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-p", PLUGIN, *args],
                          cwd=str(root), env=plugin_env(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, errors="replace", timeout=120)


@pytest.mark.parametrize("value", ["3/2", "2/2", "-1/2", "1/0", "a/b", "1", "1/2/3"])
def test_plugin_rejects_malformed_shard(tmp_path, value):
    """User runs pytest with the plugin loaded and a malformed --shard (e.g. 3/2) -> usage error, exit code 4."""
    root = make_root(tmp_path, {"tests/test_a.py": PASSING_A})
    res = run_plugin(root, f"--shard={value}", "tests/test_a.py")
    assert res.returncode == 4, res.stdout
    assert "--shard" in res.stdout, res.stdout


def test_plugin_shard_and_no_shard(tmp_path):
    """User runs pytest with the plugin loaded: no --shard -> all 3 tests run; --shard 1/2 -> only the 2nd test runs
    and the other 2 are reported as deselected."""
    root = make_root(tmp_path, {"tests/test_f.py": PASSING_A + PASSING_B})
    res = run_plugin(root, "tests/test_f.py")
    assert res.returncode == 0, res.stdout
    assert re.search(r"\b3 passed\b", res.stdout) and "deselected" not in res.stdout, res.stdout
    res = run_plugin(root, "--shard", "1/2", "-vv", "tests/test_f.py")  # -q -vv: one line per test
    assert res.returncode == 0, res.stdout
    assert re.search(r"\b1 passed, 2 deselected\b", res.stdout), res.stdout
    assert "test_two PASSED" in res.stdout and "test_one" not in res.stdout, res.stdout


# ── usage errors ─────────────────────────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("args, named", [
    (["--split", "tests/test_a.py"], "--split"),                        # no =K
    (["--split", "tests/test_a.py=0"], "--split"),                      # K < 1
    (["--split", "tests/test_a.py=x"], "--split"),                      # K not an int
    (["--split", "tests/test_nope.py=2"], "tests/test_nope.py"),        # file does not exist
    (["--split", "tests/test_studio_e2e.py=2"], "tests/test_studio_e2e.py"),  # never split (fixed port 8799)
    (["tests/test_nope.py"], "tests/test_nope.py"),                     # positional file does not exist
    (["--jobs", "0"], "--jobs"),
    (["--timeout", "0"], "--timeout"),
], ids=["split-no-k", "split-zero", "split-nan", "split-missing", "split-studio", "missing-file", "jobs0", "timeout0"])
def test_usage_errors_exit_2(tmp_path, args, named):
    """User passes a malformed option -> exit 2, a usage message naming the problem (`named`, str), nothing run
    (no log folder)."""
    root = make_root(tmp_path, {"tests/test_a.py": PASSING_A, "tests/test_studio_e2e.py": PASSING_B})
    log_dir = tmp_path / "logs"
    res = run_runner("--root", str(root), "--log-dir", str(log_dir), *args, timeout=60)
    assert res.returncode == 2, res.stdout
    assert "usage:" in res.stdout and "error:" in res.stdout, res.stdout
    error_line = next(ln for ln in res.stdout.splitlines() if "error:" in ln)  # str, argparse's error line
    assert named in error_line, res.stdout
    assert not log_dir.exists(), res.stdout


# ── CPU priority: browser E2E files normal, everything else below normal ─────────────────────────────────────────

# PRIO_TEST: str, fake test (format with out=<path>) writing its own interpreter's CPU priority to `out`:
# Windows priority class (int, 0x20 normal / 0x4000 below normal), POSIX nice value (int)
PRIO_TEST = """
    import os, pathlib

    def _priority():
        if os.name == "nt":
            import ctypes
            k32 = ctypes.windll.kernel32
            k32.GetCurrentProcess.restype = ctypes.c_void_p
            k32.GetPriorityClass.argtypes = [ctypes.c_void_p]
            k32.GetPriorityClass.restype = ctypes.c_uint32
            return k32.GetPriorityClass(k32.GetCurrentProcess())
        return os.nice(0)

    def test_record_priority():
        pathlib.Path({out!r}).write_text(str(_priority()))
"""


def expected_priorities() -> tuple[int, int]:
    """(normal, below normal) priority values (tuple[int, int]) the fake PRIO_TEST reports for a runner started by
    run_runner(normal_priority=True): Windows priority classes; POSIX this process's nice value and that + 10."""
    if os.name == "nt":
        return 0x20, 0x4000  # NORMAL_PRIORITY_CLASS, BELOW_NORMAL_PRIORITY_CLASS
    base = os.nice(0)  # int, nice value the runner inherits from this test
    return base, min(base + 10, 19)


def make_prio_root(tmp_path: Path) -> tuple[Path, Path]:
    """Root (Path) with a browser-E2E-named file tests/test_e2e_x.py and a plain tests/test_y.py, both PRIO_TEST;
    returns (root, marks folder (Path) where each writes <file stem>.txt)."""
    marks = tmp_path / "prio"
    marks.mkdir()
    root = make_root(tmp_path, {
        "tests/test_e2e_x.py": PRIO_TEST.format(out=str(marks / "test_e2e_x.txt")),
        "tests/test_y.py": PRIO_TEST.format(out=str(marks / "test_y.txt")),
    })
    return root, marks


@pytest.mark.parametrize("flat", [False, True], ids=["default", "flat-priority"])
def test_non_e2e_units_run_below_normal(tmp_path, flat):
    """User runs the runner on a root with tests/test_e2e_x.py and tests/test_y.py -> the e2e file's test ran at
    normal CPU priority, the other below normal, exit 0; with --flat-priority both ran at normal priority."""
    root, marks = make_prio_root(tmp_path)
    extra = ["--flat-priority"] if flat else []  # list[str], optional flag
    res = run_runner("--root", str(root), "--log-dir", str(tmp_path / "logs"), *extra, normal_priority=True)
    assert res.returncode == 0, res.stdout
    assert re.search(r"\b2 passed\b", last_line(res.stdout)), res.stdout
    normal, below = expected_priorities()
    got = {p.stem: int(p.read_text()) for p in marks.glob("*.txt")}  # dict[str, int], file stem -> priority seen
    assert got == {"test_e2e_x": normal, "test_y": normal if flat else below}, (got, res.stdout)


def test_list_marks_interactive_units(tmp_path):
    """User runs --list on that root -> the e2e unit is marked interactive, the other is not."""
    root, _ = make_prio_root(tmp_path)
    res = run_runner("--root", str(root), "--list", timeout=60)
    assert res.returncode == 0, res.stdout
    e2e_line = next(ln for ln in res.stdout.splitlines() if "tests/test_e2e_x.py" in ln)  # str, its --list row
    other_line = next(ln for ln in res.stdout.splitlines() if "tests/test_y.py" in ln)  # str, its --list row
    assert "interactive" in e2e_line, res.stdout
    assert "interactive" not in other_line, res.stdout
