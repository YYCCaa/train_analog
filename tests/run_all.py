"""Parallel test-suite runner: every test file (or shard of a slow one) runs in its own `python -m pytest` process,
up to --jobs at a time, longest first. Standard library only (no pytest-xdist / pytest-timeout needed).

    discover()                make_units()                    run_pool()                          summary
    tests/test_*.py      ──►  file          -> 1 unit    ──►  <= --jobs pytest processes    ──►   one line per finished
    */tests/test_*.py         file in SPLIT -> K units        longest WEIGHT first                unit, as it ends
    (or positional files)     (`-p run_all_shard            timeout -> kill the whole tree      failed units: log tail
                               --shard i/K`, plugin            stdout+stderr ─┐                   files that ran 0 tests
                               tests/run_all_shard.py)                        ▼                   last line: totals
                                                          <log-dir>/<unit>.log                    -> PASS | FAIL

    one unit = [python, -m, pytest, -q, -p, no:cacheprovider, (-p, run_all_shard, --shard, i/K,) FILE], cwd = root,
               OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS = --threads (unless already set), stdin = /dev/null

Usage (from the repo root):
    python tests/run_all.py                         # whole suite
    python tests/run_all.py --list                  # show the units in scheduled order, run nothing
    python tests/run_all.py tests/test_v7_render.py # only that file (still split into its shards)
    python tests/run_all.py --split tests/test_v7_render.py=1 --jobs 4 --timeout 900

Exit code: 0 every unit passed, 1 anything failed / timed out / a file ran no tests / Ctrl+C, 2 usage error.
"""

from __future__ import annotations

import argparse
import locale
import os
import posixpath
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import IO, Callable

# SPLIT: dict[str, int], test file (root-relative posix path) -> number of shards K it runs as (K units)
SPLIT = {
    "tests/test_v7_render.py": 8,
    "val_viewer/tests/test_api.py": 4,
    "tests/test_export_v9_tflite.py": 6,
    "tests/test_train_v9.py": 3,
    "tests/test_v7_gray_generator.py": 2,
    "tip_webapp/tests/test_e2e_browser.py": 2,
    "val_viewer/tests/test_e2e_browser.py": 2,
}
# NEVER_SPLIT: frozenset[str], files that must run as one process (fixed port 8799; its server may build the frontend)
NEVER_SPLIT = frozenset({"tests/test_studio_e2e.py"})
# WEIGHT: dict[str, float], expected seconds per test file (root-relative posix path) when run alone, whole
WEIGHT = {
    "tests/test_v7_render.py": 290,
    "val_viewer/tests/test_api.py": 130,
    "tests/test_export_v9_tflite.py": 90,
    "tests/test_studio_e2e.py": 75,
    "tests/test_train_v9.py": 65,
    "tests/test_train_gray.py": 48,
    "val_viewer/tests/test_e2e_browser.py": 40,
    "tip_webapp/tests/test_e2e_browser.py": 22,
    "tests/test_v7_look.py": 15,
    "tests/test_v7_gray_generator.py": 15,
    "tests/test_studio_api.py": 8,
}
# DEFAULT_WEIGHT: float, expected seconds of a file missing from WEIGHT
DEFAULT_WEIGHT = 30.0
# THREAD_VARS: tuple[str, ...], native thread-pool sizes capped by --threads in the children's environment
THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
# PLUGIN_MODULE: str, importable name of the shard plugin (unique, so no other `tests` package can shadow it)
PLUGIN_MODULE = "run_all_shard"
# PLUGIN_DIR: Path, folder holding this file and the plugin; prepended to PYTHONPATH of sharded units
PLUGIN_DIR = Path(__file__).resolve().parent
# DEFAULT_ROOT: Path, project root searched when --root is not given (parent of the folder holding this file)
DEFAULT_ROOT = PLUGIN_DIR.parent
# DISCOVER_PATTERNS: tuple[str, ...], glob patterns (relative to root) of the test files that make up the suite
DISCOVER_PATTERNS = ("tests/test_*.py", "*/tests/test_*.py")
# TAIL_LINES: int, how many last log lines are printed for each failed / timed-out unit
TAIL_LINES = 40
# TAIL_BYTES: int, how many bytes at the end of a log are read to find its summary line and tail
TAIL_BYTES = 256 * 1024
# POLL_S: float, seconds between two checks of the running processes
POLL_S = 0.05
# LOG_MARK: str, prefix of the lines the runner itself writes into a unit's log (header / footer)
LOG_MARK = "# run_all:"
# LOG_ENCODING: str, encoding a child's redirected stdout uses (the locale's), also used for our header / footer
LOG_ENCODING = locale.getpreferredencoding(False)
# COUNT_RE: re.Pattern, one "<n> <outcome>" item of pytest's final summary line
COUNT_RE = re.compile(r"(\d+) (passed|failed|skipped|errors?|xfailed|xpassed|deselected)\b")
# SUMMARY_TIME_RE: re.Pattern, the " in 12.34s" ending of pytest's final summary line
SUMMARY_TIME_RE = re.compile(r"\bin \d+(?:\.\d+)?s\b")
# COUNT_KEYS: tuple[str, ...], outcome counters shown per unit, in display order
COUNT_KEYS = ("passed", "failed", "skipped", "errors", "xfailed", "xpassed")


# ── units ────────────────────────────────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Unit:
    """One pytest process: a whole test file (k == 1) or shard `shard` of `k` of it."""
    file: str            # str, test file path relative to root (posix)
    shard: int | None    # int | None, 0-based shard index (the plugin's --shard i), None for a whole file
    k: int               # int, number of shards the file is split into (1 = whole file)
    weight: float        # float, expected seconds (file weight / k), the scheduling key

    @property
    def name(self) -> str:
        """Filesystem-safe unique name (str), e.g. `tests__test_v7_render.py__3of8` (shard 2/8, 1-based in names)."""
        base = self.file.replace("/", "__")
        return base if self.shard is None else f"{base}__{self.shard + 1}of{self.k}"

    @property
    def display(self) -> str:
        """Human label (str), e.g. `tests/test_v7_render.py [3/8]` (1-based shard number of K)."""
        return self.file if self.shard is None else f"{self.file} [{self.shard + 1}/{self.k}]"


@dataclass
class Result:
    """Outcome of one finished (or killed) unit."""
    unit: Unit                 # Unit, what ran
    status: str                # str, "PASS" (exit 0) | "EMPTY" (exit 5, nothing collected) | "FAIL" | "TIMEOUT" | "INTERRUPTED"
    exit_code: int | None      # int | None, pytest's exit code (None when the runner killed it)
    seconds: float             # float, wall-clock seconds the unit ran
    counts: dict[str, int]     # dict[str, int], pytest summary counters (passed, failed, skipped, errors, ...)
    log: Path                  # Path, the unit's log file

    @property
    def ok(self) -> bool:
        """True (bool) when this unit by itself does not fail the run (PASS, or EMPTY: an empty shard is fine)."""
        return self.status in ("PASS", "EMPTY")

    @property
    def ran(self) -> int:
        """Number of tests that actually ran (int): passed + failed + skipped + errors + xfailed + xpassed."""
        return sum(self.counts.get(key, 0) for key in COUNT_KEYS)


@dataclass
class Running:
    """A unit whose pytest process is alive."""
    unit: Unit                 # Unit, what runs
    proc: subprocess.Popen     # subprocess.Popen, the `python -m pytest` process (on Windows: the venv launcher)
    log_fh: IO[bytes]          # IO[bytes], the log file the process writes stdout+stderr into
    log: Path                  # Path, that log file's path
    t0: float = field(default_factory=time.monotonic)  # float, time.monotonic() at start


def discover(root: Path) -> list[str]:
    """Every test file of the suite under `root` (Path): DISCOVER_PATTERNS, skipping top-level hidden folders
    (.venv, .git, .claude). Returns sorted root-relative posix paths (list[str])."""
    found = set()  # set[str], root-relative posix paths
    for pattern in DISCOVER_PATTERNS:
        for path in root.glob(pattern):
            rel = path.relative_to(root).as_posix()
            if path.is_file() and not rel.startswith("."):
                found.add(rel)
    return sorted(found)


def make_units(files: list[str], split: dict[str, int]) -> list[Unit]:
    """Turn test files (list[str], root-relative) into units (list[Unit]) in scheduled order: longest weight first,
    ties by name. A file with split[file] = K > 1 becomes K shards, each weighing WEIGHT/K."""
    units = []  # list[Unit]
    for f in files:
        k = 1 if f in NEVER_SPLIT else max(1, split.get(f, 1))  # int, shards of this file
        w = WEIGHT.get(f, DEFAULT_WEIGHT) / k  # float, expected seconds of one shard
        units += [Unit(f, None, 1, w)] if k == 1 else [Unit(f, i, k, w) for i in range(k)]
    return sorted(units, key=lambda u: (-u.weight, u.name))


# ── command line ─────────────────────────────────────────────────────────────────────────────────────────────────

def parse_split(text: str) -> tuple[str, int]:
    """Parse a --split value `text` (str, "FILE=K") into (FILE, K) (tuple[str, int], K >= 1); raises
    argparse.ArgumentTypeError on a malformed value."""
    file, sep, k_text = text.rpartition("=")
    if not sep or not file:
        raise argparse.ArgumentTypeError(f"expected FILE=K, got {text!r}")
    try:
        k = int(k_text)  # int, requested shard count
    except ValueError:
        raise argparse.ArgumentTypeError(f"K must be an integer in {text!r}") from None
    if k < 1:
        raise argparse.ArgumentTypeError(f"K must be >= 1 in {text!r} (1 = do not split)")
    return file, k


def build_parser() -> argparse.ArgumentParser:
    """The command-line parser (argparse.ArgumentParser)."""
    p = argparse.ArgumentParser(
        prog="run_all.py", description="Run the test suite as parallel pytest processes (one per file / shard).")
    p.add_argument("files", nargs="*", metavar="FILE",
                   help="test files to run (relative to --root); default: every tests/test_*.py and */tests/test_*.py")
    p.add_argument("--root", default=None, help=f"project root (default: {DEFAULT_ROOT})")
    p.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2),
                   help="max units running at once (default: CPU count // 2)")
    p.add_argument("--timeout", type=float, default=600.0,
                   help="seconds per unit before its whole process tree is killed (default 600)")
    p.add_argument("--threads", type=int, default=4,
                   help="OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS for each unit unless already set (0 = leave alone)")
    p.add_argument("--split", action="append", type=parse_split, default=[], metavar="FILE=K",
                   help="run FILE as K round-robin shards (repeatable; K=1 = whole; overrides the built-in table)")
    p.add_argument("--log-dir", default=None,
                   help="folder for the per-unit logs (default: <root>/tmp_smoke/testlogs/run_all/<YYYYMMDD-HHMMSS>)")
    p.add_argument("--list", action="store_true", help="print the units in scheduled order and exit")
    return p


def to_rel(root: Path, text: str) -> str | None:
    """Root-relative posix path (str) of the user-given path `text` (str, relative to root or absolute);
    None when it lies outside `root` (Path)."""
    path = Path(text)
    try:
        rel = os.path.relpath(path, root) if path.is_absolute() else text  # str, path relative to root
    except ValueError:  # Windows: another drive
        return None
    rel = posixpath.normpath(rel.replace("\\", "/"))
    return None if rel == ".." or rel.startswith("../") or os.path.isabs(rel) else rel


# ── processes ────────────────────────────────────────────────────────────────────────────────────────────────────

def child_envs(threads: int) -> tuple[dict[str, str], dict[str, str]]:
    """Environments (dict[str, str]) for whole-file units and for sharded units: the runner's environment with
    THREAD_VARS set to `threads` (int, 0 = untouched) where the runner has not set them; the sharded one also has
    PLUGIN_DIR prepended to PYTHONPATH so `-p run_all_shard` imports whatever --root is."""
    env = dict(os.environ)
    if threads > 0:
        for var in THREAD_VARS:
            env.setdefault(var, str(threads))
    sharded = dict(env)
    old = sharded.get("PYTHONPATH")
    sharded["PYTHONPATH"] = str(PLUGIN_DIR) + (os.pathsep + old if old else "")
    return env, sharded


def unit_cmd(unit: Unit) -> list[str]:
    """The pytest command line (list[str]) of `unit` (Unit)."""
    cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"]
    if unit.shard is not None:
        cmd += ["-p", PLUGIN_MODULE, "--shard", f"{unit.shard}/{unit.k}"]
    return cmd + [unit.file]


def start_unit(unit: Unit, root: Path, log_dir: Path, env: dict[str, str]) -> Running:
    """Start `unit` (Unit) in `root` (Path) with environment `env` (dict[str, str]), output to
    `log_dir`/<unit name>.log (a header naming the command comes first). Returns the Running handle."""
    log = log_dir / f"{unit.name}.log"
    cmd = unit_cmd(unit)
    log_fh = open(log, "wb")
    log_fh.write(f"{LOG_MARK} unit {unit.name} ({unit.display})\n{LOG_MARK} cwd {root}\n"
                 f"{LOG_MARK} cmd {subprocess.list2cmdline(cmd)}\n".encode(LOG_ENCODING, "replace"))
    log_fh.flush()
    if os.name == "nt":  # own process group: the console's Ctrl+C reaches only the runner, which kills the trees
        extra = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    else:                # own session = own process group, killed as a whole with os.killpg
        extra = {"start_new_session": True}
    try:
        proc = subprocess.Popen(cmd, cwd=str(root), env=env, stdin=subprocess.DEVNULL, stdout=log_fh,
                                stderr=subprocess.STDOUT, **extra)
    except BaseException:
        log_fh.close()
        raise
    return Running(unit, proc, log_fh, log)


def kill_tree(proc: subprocess.Popen) -> None:
    """Kill `proc` (subprocess.Popen) and every process below it, then reap it. On Windows `taskkill /F /T` (the
    venv python.exe is a launcher whose child is the real interpreter, so the top PID alone is not enough);
    elsewhere SIGKILL to its process group."""
    if proc.poll() is None:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], stdin=subprocess.DEVNULL,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60)
        else:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=30)


def read_log_tail(log: Path) -> str:
    """The last TAIL_BYTES of `log` (Path) as text (str): UTF-8 if it decodes, else the locale encoding."""
    try:
        with open(log, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            fh.seek(max(0, fh.tell() - TAIL_BYTES))
            data = fh.read()  # bytes, end of the log
    except OSError:
        return ""
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode(LOG_ENCODING, "replace")


def parse_counts(text: str) -> dict[str, int]:
    """Counters (dict[str, int]: passed, failed, skipped, errors, xfailed, xpassed, deselected) from pytest's final
    summary line in `text` (str, log tail), e.g. "3 passed, 1 skipped in 1.20s"; {} when there is none."""
    for line in reversed(text.splitlines()):
        line = line.strip().strip("=").strip()
        if not SUMMARY_TIME_RE.search(line) or line.startswith(LOG_MARK):
            continue
        items = COUNT_RE.findall(line)  # list[tuple[str, str]], (number, outcome word)
        if items or line.startswith("no tests ran"):
            counts = {}  # dict[str, int]
            for n, word in items:
                key = "errors" if word.startswith("error") else word
                counts[key] = counts.get(key, 0) + int(n)
            return counts
    return {}


def finish(run: Running, status: str | None, timeout: float) -> Result:
    """Close `run` (Running) and turn it into a Result. `status` (str | None): None = the process exited by itself
    (status from its exit code), else "TIMEOUT" / "INTERRUPTED" (it was killed). `timeout` (float, s) is quoted in the
    log footer. Appends a footer line `# run_all: <status> ...` to the log."""
    seconds = time.monotonic() - run.t0  # float, wall-clock run time
    run.log_fh.close()
    code = run.proc.returncode if status is None else None  # int | None, pytest's exit code
    if status is None:
        status = {0: "PASS", 5: "EMPTY"}.get(code, "FAIL")
    counts = parse_counts(read_log_tail(run.log))
    note = f"exit {code}" if code is not None else (
        f"killed after {timeout:g} s (process tree)" if status == "TIMEOUT" else "killed (Ctrl+C)")
    try:
        with open(run.log, "ab") as fh:
            fh.write(f"\n{LOG_MARK} {status} {note} in {seconds:.1f} s\n".encode(LOG_ENCODING, "replace"))
    except OSError:
        pass
    return Result(run.unit, status, code, seconds, counts, run.log)


def run_pool(units: list[Unit], root: Path, log_dir: Path, jobs: int, timeout: float,
             envs: tuple[dict[str, str], dict[str, str]], on_done: Callable[[Result], None]) -> tuple[list[Result], bool]:
    """Run `units` (list[Unit], scheduled order) with at most `jobs` (int) alive at once; a unit running longer than
    `timeout` (float, s) has its process tree killed (TIMEOUT). `envs`: (whole-file env, sharded env). `on_done`
    (callable(Result)) is called as each unit ends. On Ctrl+C every running tree is killed. Returns (results:
    list[Result] in finishing order, interrupted: bool)."""
    pending = list(units)  # list[Unit], not started yet (scheduled order)
    running = []           # list[Running], alive processes
    results = []           # list[Result], finished units
    try:
        while pending or running:
            while pending and len(running) < jobs:
                unit = pending.pop(0)
                running.append(start_unit(unit, root, log_dir, envs[unit.shard is not None]))
            time.sleep(POLL_S)
            for run in list(running):
                if run.proc.poll() is not None:
                    status = None
                elif time.monotonic() - run.t0 > timeout:
                    kill_tree(run.proc)
                    status = "TIMEOUT"
                else:
                    continue
                running.remove(run)  # first, so a Ctrl+C from here on cannot report this unit twice
                res = finish(run, status, timeout)
                results.append(res)
                on_done(res)
    except KeyboardInterrupt:
        finished = {r.unit for r in results}  # set[Unit], units already reported
        left = [run for run in running if run.unit not in finished]  # list[Running], units to kill
        for run in left:
            kill_tree(run.proc)
        for run in left:
            res = finish(run, "INTERRUPTED", timeout)
            results.append(res)
            on_done(res)
        return results, True
    except BaseException:  # the runner itself broke (e.g. Popen failed): leave no orphaned pytest behind
        for run in running:
            kill_tree(run.proc)
        raise
    return results, False


# ── reporting ────────────────────────────────────────────────────────────────────────────────────────────────────

def counts_text(res: Result, timeout: float) -> str:
    """Short outcome text (str) of `res` (Result), e.g. "14 passed, 1 skipped" / "no tests ran" / "timed out ..."."""
    if res.status == "TIMEOUT":
        return f"timed out after {timeout:g} s, process tree killed"
    if res.status == "INTERRUPTED":
        return "interrupted, process tree killed"
    parts = [f"{res.counts[k]} {k}" for k in COUNT_KEYS if res.counts.get(k)]  # list[str], nonzero counters
    if res.status == "EMPTY" and not parts:
        return "no tests ran"
    if res.status == "FAIL" and res.exit_code not in (1, None):
        parts.append(f"exit {res.exit_code}")
    return ", ".join(parts) or f"exit {res.exit_code}"


def tail_lines(log: Path, n: int = TAIL_LINES) -> list[str]:
    """The last `n` (int) lines (list[str]) of `log` (Path), without the runner's own footer."""
    lines = [ln for ln in read_log_tail(log).splitlines() if not ln.startswith(LOG_MARK)]
    while lines and not lines[-1].strip():
        lines.pop()
    return lines[-n:]


def summarize(results: list[Result], units: list[Unit], wall: float, jobs: int, timeout: float,
              interrupted: bool) -> bool:
    """Print the failure section and the final line; returns True (bool) when the whole run passed.
    `results` (list[Result]) finished units, `units` (list[Unit]) every scheduled unit, `wall` (float) seconds."""
    bad = [r for r in results if not r.ok]  # list[Result], failed / timed-out / interrupted units
    by_file = {}  # dict[str, list[Result]], test file -> results of its units
    for r in results:
        by_file.setdefault(r.unit.file, []).append(r)
    expected = {}  # dict[str, int], test file -> number of units scheduled for it
    for u in units:
        expected[u.file] = expected.get(u.file, 0) + 1
    no_tests = [f for f, rs in sorted(by_file.items())   # list[str], files whose units all finished OK yet ran 0 tests
                if len(rs) == expected[f] and all(r.ok for r in rs) and sum(r.ran for r in rs) == 0]
    if bad or no_tests:
        print("\n" + "=" * 30 + " failures " + "=" * 30, flush=True)
    for r in sorted(bad, key=lambda r: r.unit.name):
        note = f"exit {r.exit_code}" if r.exit_code is not None else r.status.lower()
        print(f"\n---- {r.status} {r.unit.name} ({r.unit.display}, {note})  log: {r.log}")
        for line in tail_lines(r.log) or ["(the unit printed nothing)"]:
            print(f"    {line}")
    for f in no_tests:
        n = expected[f]
        logs = ", ".join(str(r.log) for r in by_file[f])
        print(f"\n---- FAIL {f}: no tests (its {n} unit{'s' if n > 1 else ''} ran 0 tests)  log: {logs}")
    tot = {k: sum(r.counts.get(k, 0) for r in results) for k in ("passed", "failed", "skipped", "errors")}
    not_started = len(units) - len(results)  # int, units never started (Ctrl+C)
    extra = []  # list[str], what made the run fail besides failing tests
    for status in ("FAIL", "TIMEOUT", "INTERRUPTED"):
        n = sum(r.status == status for r in results)
        if n:
            extra.append(f"{n} {status}")
    if no_tests:
        extra.append(f"{len(no_tests)} file{'s' if len(no_tests) > 1 else ''} with no tests")
    if not_started:
        extra.append(f"{not_started} not started")
    passed = not bad and not no_tests and not interrupted and not not_started  # bool, overall verdict
    print(f"\nrun_all: {tot['passed']} passed, {tot['failed']} failed, {tot['skipped']} skipped, {tot['errors']} errors"
          f" in {wall:.1f} s ({len(units)} units, jobs {jobs}{''.join('; ' + e for e in extra)})"
          f" -> {'PASS' if passed else 'FAIL'}", flush=True)
    return passed


def default_log_dir(root: Path) -> Path:
    """A fresh <root>/tmp_smoke/testlogs/run_all/<YYYYMMDD-HHMMSS>[-n] folder path (Path, not created)."""
    base = root / "tmp_smoke" / "testlogs" / "run_all" / datetime.now().strftime("%Y%m%d-%H%M%S")
    path, n = base, 1  # Path, candidate; int, suffix counter
    while path.exists():
        n += 1
        path = base.with_name(f"{base.name}-{n}")
    return path


def main(argv: list[str] | None = None) -> int:
    """Command-line entry: parse `argv` (list[str] | None = sys.argv[1:]), run or list the units; returns the exit
    code (int: 0 pass, 1 fail, 2 usage error — argparse exits with 2 itself)."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")  # log tails may hold characters the console cannot show
    parser = build_parser()     # argparse.ArgumentParser
    args = parser.parse_args(argv)  # argparse.Namespace: files, root, jobs, timeout, threads, split, log_dir, list
    root = Path(args.root).resolve() if args.root else DEFAULT_ROOT  # Path, project root
    if not root.is_dir():
        parser.error(f"--root {root}: not a folder")
    if args.jobs < 1:
        parser.error(f"--jobs must be >= 1, got {args.jobs}")
    if not args.timeout > 0:
        parser.error(f"--timeout must be > 0 s, got {args.timeout:g}")
    if args.threads < 0:
        parser.error(f"--threads must be >= 0, got {args.threads}")

    split = dict(SPLIT)  # dict[str, int], built-in table + --split overrides
    for file, k in args.split:
        rel = to_rel(root, file)
        if rel is None or not (root / rel).is_file():
            parser.error(f"--split {file}: no such file under {root}")
        if rel in NEVER_SPLIT and k > 1:
            parser.error(f"--split {rel}={k}: {rel} must run as one process (fixed port 8799)")
        split[rel] = k

    if args.files:
        files = []  # list[str], root-relative test files to run
        for file in args.files:
            rel = to_rel(root, file)
            if rel is None or not (root / rel).is_file():
                parser.error(f"{file}: no such file under {root}")
            if rel not in files:
                files.append(rel)
    else:
        files = discover(root)
        if not files:
            parser.error(f"no tests/test_*.py or */tests/test_*.py under {root}")
    units = make_units(files, split)  # list[Unit], in scheduled (longest-first) order

    if args.list:
        print(f"run_all --list: {len(units)} units from {len(files)} files, jobs {args.jobs}, root {root}")
        width = max(len(u.display) for u in units)  # int, label column width
        for u in units:
            print(f"{u.weight:8.1f} s  {u.display:<{width}}  ({u.name})")
        return 0

    log_dir = Path(os.path.abspath(args.log_dir)) if args.log_dir else default_log_dir(root)  # Path, per-unit logs
    log_dir.mkdir(parents=True, exist_ok=True)
    envs = child_envs(args.threads)  # tuple[dict[str, str], dict[str, str]], (whole-file env, sharded env)
    print(f"run_all: {len(units)} units from {len(files)} files, jobs {args.jobs}, timeout {args.timeout:g} s, "
          f"threads {args.threads or 'unchanged'}, logs {log_dir}", flush=True)
    width = len(str(len(units)))  # int, digits of the unit counter
    done = [0]  # list[int], finished-unit counter (mutable for the callback)

    def on_done(res: Result) -> None:
        """Print one line for the finished unit `res` (Result)."""
        done[0] += 1
        print(f"[{done[0]:>{width}}/{len(units)}] {res.status:<7} {res.seconds:6.1f}s  {res.unit.display}  "
              f"{counts_text(res, args.timeout)}", flush=True)

    t0 = time.monotonic()  # float, start of the run
    results, interrupted = run_pool(units, root, log_dir, args.jobs, args.timeout, envs, on_done)
    if interrupted:
        print("run_all: interrupted (Ctrl+C), every running unit's process tree was killed", flush=True)
    passed = summarize(results, units, time.monotonic() - t0, args.jobs, args.timeout, interrupted)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
