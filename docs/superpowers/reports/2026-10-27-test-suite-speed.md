# Test suite speed-up: from ~5 min (7 hand-made groups) to ~1.5 min (one command)

Goal (user, 2026-10-27): "optimize/cut entire test suite to under 3 minutes".

Result: `python tests/run_all.py` runs all 22 test files (506 tests) in **82–99 s** when the machine is otherwise
quiet. It took **143–157 s** while another Claude session's suites, a game and Defender were busy. Both are under
the 3-minute target. The final `WT verify` at 5cc6a8a: `506 passed, 0 failed, 0 skipped, 0 errors in 99.3 s
(40 units, jobs 16) -> PASS`.

## 1. Before and after

Baseline (main 066b49a, 20:33): the suite had no runner. The fastest way to run it was the 7 groups below, started by
hand as parallel `pytest` processes. One test (the v4 parity test) always failed.

| group (baseline) | tests | wall time |
|---|---|---|
| v7 render + v7 gray generator | 118 | 295.7 s |
| val_viewer API + browser E2E | 55 | 189.3 s |
| v9 training + TFLite export | 28 | 159.7 s |
| studio API + browser E2E | 64 | 81.6 s |
| gray training + autocal + v4 | 96 (1 failed) | 53.5 s |
| tip_webapp + grayscale_webapp | 111 | 30.9 s |
| v7 look-match | 1 | 14.8 s |
| **sum (= one serial pytest)** | **473** | **~825 s (13.7 min)** |
| **wall (7 groups at once)** | | **295.7 s, FAIL** |

```
before   v7 group     |█████████████████████████████████████████████████████████████| 295.7 s  (one 103 s test inside)
         val_viewer   |██████████████████████████████████████|                         189.3 s
         v9 train+exp |████████████████████████████████|                               159.7 s
         studio       |████████████████|                                                81.6 s
         ...          |███████████|                                                     <= 53.5 s
                      0 s                    60 s                    120 s        180 s (goal)       300 s

after    run_all.py   |████████████████████|  82-99 s quiet          (143-157 s under heavy outside load)
                      40 processes, 16 at a time
```

After: 506 tests, because 30 runner tests were added and the one v4 parity test became 4 v4 tests. Slowest units in the final
verify run (a unit is one pytest process; `[i/K]` is shard i of K):

| unit | wall | | unit | wall |
|---|---|---|---|---|
| test_v7_render.py [6/8] | 78.5 s | | val_viewer test_api.py [2/2] | 52.3 s |
| test_export_v9_tflite.py [4/6] | 73.8 s | | test_v7_render.py [1/8] | 48.0 s |
| test_studio_e2e.py | 72.1 s | | val_viewer test_api.py [1/2] | 45.8 s |
| test_train_gray.py | 71.4 s | | val_viewer test_e2e_browser.py [1/2] | 36.7 s |
| test_export_v9_tflite.py [5/6] | 67.7 s | | test_run_all.py | 23.9 s |
| test_train_v9.py [3/3] | 63.7 s | | tip_webapp test_inference.py | 18.8 s |

The wall time is now set by how much CPU the whole machine has, not by one long file: 16 processes share 32 hardware
threads, and the CPU-bound tests (rendering, training, TFLite conversion) use up what is left.

## 2. The runner: `tests/run_all.py` (+ `tests/run_all_shard.py`)

pytest-xdist and pytest-timeout are not installed (and nothing new goes into the shared venv), so the runner uses
only the standard library. It starts plain `python -m pytest` processes:

```
 discover                     split (SPLIT table)                 order (WEIGHT table)        run (--jobs 16)
 tests/test_*.py      ──►   test_v7_render.py  ─► 8 shards  ──►   longest expected     ──►   ┌ slot 1  ─ unit ─ unit ┐
 */tests/test_*.py          test_export_v9...  ─► 6 shards        first, so the long         │ slot 2  ─ unit ─ ...  │
 (22 files)                 test_train_v9.py   ─► 3 shards        ones never start last      │  ...                  │
                            4 more files       ─► 2 shards each                              └ slot 16 ─ unit        ┘
                            15 files           ─► 1 unit each      (40 units)                 each: own log, own timeout
```

- **Shards** (`-p run_all_shard --shard i/K`): the plugin keeps every K-th collected test (`index % K == i`) and
  deselects the rest. Round-robin spreads a file's slow tests over its shards:

  ```
  collected:  t0  t1  t2  t3  t4  t5  t6  t7  t8  ...
  shard 0/3:  t0          t3          t6          ...
  shard 1/3:      t1          t4          t7      ...
  shard 2/3:          t2          t5          t8  ...
  ```

- **Weights:** the expected seconds per file, from measured runs; each shard gets weight / K and unlisted files get 30 s.
- **Threads:** `--threads 4` sets OMP / MKL / OPENBLAS / NUMEXPR thread counts, so 16 processes don't each start
  32 BLAS threads.
- **Timeout:** `--timeout` (default 600 s) kills a hung unit's whole process tree (`taskkill /F /T`).
- **Logs:** one log per unit in `tmp_smoke/testlogs/run_all/<timestamp>/`. The run ends with one line,
  `run_all: ... -> PASS | FAIL`.
- **Options:** `--list` (show the plan), positional files, `--split FILE=K`, `--jobs`, `--flat-priority`.

### Priority split: why browser tests run at normal priority

The first parallel runs failed now and then in the browser E2E files. Their app servers and Chromium were starved by
the CPU-bound tests, and Playwright's waits (page load 15 s, `expect` 5 s) ran out. Raising those timeouts would
only have hidden the problem. Instead, the runner starts every file whose name contains `e2e` at normal priority and
every other file at below-normal priority (POSIX: `nice +10`):

```
 CPU time goes first to ──►  [ normal ]       studio / tip / grayscale / val_viewer E2E: servers + Chromium answer quickly
 then what is left    ──►  [ below normal ]  render / training / export: they soak up every idle cycle, nothing is lost
```

No browser flake has appeared since this change, and the total time didn't grow, because the low-priority units still
get all the idle CPU.

## 3. Changes in the slow test files

| file | before | change | effect |
|---|---|---|---|
| tests/test_v7_render.py | one test took 103 s (2000 seeds, serial) | split into `..._headroom_rate` and `..._fallback_rate` (same n = 2000, same assertions); each computes its pointer flags in a 4-worker process pool over interleaved seed chunks | the slowest single test is gone; the file runs as 8 shards |
| tests/test_export_v9_tflite.py | every CLI test paid for a full TFLite conversion | module fixture `onnx_only` exports ONNX once; `exported` converts on top of it only for the tests that need TFLite | less repeated work; 6 shards |
| val_viewer/tests/test_api.py + conftest.py | every test built a 200-image fake dataset and ranked it (~5 s wall / 12 CPU-s per ranking) | 11 read-only tests share one module-scoped client on the 200-image set; 17 tests that need their own server use a 40-image set (`small_fake_data`, 34 train / 6 val), with sizes derived from `n` | ~512 → ~140 CPU-s |
| tests/test_v4_generator.py | `sample_params matches v3` always failed (v4 diverged from v3 on purpose in 3da29e3) | replaced (user's choice) by: v4 range test, bezel-fit test, shared-range test, strict-xfail full-promise test | suite is green; see §5 |

## 4. A scheduling experiment that was reverted

a4d081d tried "real" weights for every file. It raised export to 240 and train_v9 to 135, which made all their shards
start in the first wave. The verify run took 157 s, against 143.5 s for the old table under similar outside load,
because the heavy units then all competed at the same moment. It was reverted (5cc6a8a). The kept table starts some
light files first and lets the heavy shards fill in, and it gave the 99.3 s final verify.

## 5. Finding: v4's centre clamp lets the bezel overflow the image

To keep the dial inside the 256 px image, `generate_synthetic_v4.sample_params` clamps the centre using the box
`|A cos θ| + |B sin θ|`. That box is wider than the true half-width of a rotated ellipse,
`sqrt(A² cos² θ + B² sin² θ)`. When the box is wider than the image, the clamp's bounds cross, the centre is pushed
off the middle, and the true bezel then sticks out:

```
        image 0 ────────────────────────────── 256
 clamp box   [··························································]  wider than the image: the bounds cross
 true bezel     (──────────────────────────────────────)            would fit if centred ...
 as sampled          (──────────────────────────────────────)       ... but the centre was pushed: overflow up to 21.5 px
```

The bezel overflows on 131 of 2000 seeds, by up to 21.5 px, and every one of those dials would fit if centred. The
new tests pin the regime where the clamp works (`..._when_clamp_box_fits`). `test_sample_params_bezel_always_inside_image`
records the full promise as `xfail(strict=True)`: once the clamp is fixed, it XPASSes, the run fails, and the marker
can be removed. Fixing the generator is a separate decision. Also, v4's `A_RANGE` is narrower than v3's (all
other ranges are shared). An earlier note in this session said every range was shared; that was wrong.

## 6. Timings depend on outside load

| run | load | time |
|---|---|---|
| quiet machine (several runs) | only this suite | 82–95 s |
| final `WT verify`, 5cc6a8a | the other session paused; game in the background | 99.3 s |
| comparison run, same tree | game, Defender and the search indexer busy | 143.5 s |
| verify at a4d081d (reverted table) | similar | 157.0 s |

Before timing the suite, wait until no other `run_all.py` is running, and ask any other session to pause its suites.

## 7. How to run

```
python tests/run_all.py                          # whole suite, ~1.5 min
python tests/run_all.py --list                   # show the 40 units, priorities and weights without running
python tests/run_all.py tests/test_v4_generator.py val_viewer/tests/test_api.py   # only these files
python C:/Users/Administrator/.claude/hooks/wt.py verify    # same command via .wtflow.json, verify_timeout 900 s
```

Plain `python -m pytest` still works and runs serially (~11 min). The test map in CLAUDE.md lists per-area commands
and which files are not parallel-safe: the studio E2E binds port 8799 and is never split; v9 training and gray
training use fixed folders. That is why only one `run_all.py` may run per checkout at a time.
