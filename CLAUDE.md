# train_dial

Analog-dial autocalibration: synthetic dial generators (v3..v7), tip / ellipse models, TFLite export, and four
local web apps (`tip_webapp/`, `grayscale_webapp/`, `studio/`, and `val_viewer/`: best / worst n TipNet v9
images of the val / train / all set with heatmaps, plus each image's saved parameters when the dataset was
generated with `--save-params` / the studio's "save parameters"). Data and training runs live outside the repo in
`../train_dial_data` (`sv9_150k`, `runs/`).

## Tests

Logs: `tmp_smoke/testlogs/<task>/<timestamp>/` (gitignored), written by
`python C:/Users/Administrator/.claude/hooks/wt.py testlog <task> <label> -- <command>`.
Under `testlog`, a bare `python` resolves to the system Python312 (no pytest), and the worktree guard blocks the venv
interpreter's absolute path in agent worktrees: wrap the command in cmd so PATH picks the venv,
`... testlog <task> <label> -- cmd /c python -m pytest -q <file>` (from Git Bash write `cmd //c`).

Run every command from the repo root. No test needs a GPU, the network, or the real dataset; datasets are synthetic
and written to tmp_path / tempfile.

Whole suite: `python tests/run_all.py` (under `testlog`: `... testlog <task> suite -- cmd /c python tests/run_all.py`).
It runs every `tests/test_*.py` and `*/tests/test_*.py` as parallel `python -m pytest` processes (16 at a time, slow
files split into round-robin shards by its `SPLIT` table, longest first by its `WEIGHT` table), one log per process
in `tmp_smoke/testlogs/run_all/<timestamp>/`, and ends with one line `run_all: ... in <s> s ... -> PASS | FAIL`.
`--list` shows the processes without running them; positional files run only those; `--split FILE=K`, `--jobs N`,
`--timeout S`, `--threads T` override the defaults. Plain `python -m pytest` still works but runs serially (~11 min).
`.wtflow.json` makes `WT verify` run the whole suite this way.

| area | source files | test command | timeout |
|---|---|---|---|
| v7 config parse/validate | v7_config.py, tests/data/*.yaml | `python -m pytest -q tests/test_v7_config.py` | 60 s |
| v7 renderer | generate_synthetic_v7_mp.py, v7_config.py | `python -m pytest -q tests/test_v7_render.py` | 600 s |
| v7 generator CLI | generate_synthetic_v7_mp.py, v7_config.py | `python -m pytest -q tests/test_v7_gray_generator.py` | 600 s |
| v7 look-match (slow) | tools/compare_v7_look.py, generate_synthetic_v7_mp.py | `python -m pytest -q tests/test_v7_look.py` | 1200 s |
| v4 generator | generate_synthetic_v4.py, generate_synthetic_v3.py | `python -m pytest -q tests/test_v4_generator.py` | 60 s |
| autocal math | autocal_utils.py | `python -m pytest -q tests/test_autocal_math.py` | 120 s |
| autocal models | autocal_utils.py | `python -m pytest -q tests/test_autocal_models.py` | 120 s |
| autocal dataset | autocal_utils.py | `python -m pytest -q tests/test_autocal_dataset.py` | 120 s |
| v9 TipNet training | train_v9_tip.py, generate_synthetic_v7_mp.py, v7_config.py | `python -m pytest -q tests/test_train_v9.py` | 1200 s |
| v9 TFLite export | export_v9_tflite.py, train_v9_tip.py | `python -m pytest -q tests/test_export_v9_tflite.py` | 1200 s |
| gray InvResNet training | train_e_inv_res_gray.py, train_utils_v2_gray.py | `python -m pytest -q tests/test_train_gray.py` | 900 s |
| studio API | studio/backend/*.py, generate_synthetic_v7_mp.py | `python -m pytest -q tests/test_studio_api.py` | 600 s |
| studio browser E2E | studio/run_studio.py, studio/backend/*, studio/frontend | `python -m pytest -q tests/test_studio_e2e.py` | 1200 s |
| tip_webapp inference | tip_webapp/app/inference.py, tip_webapp/model/, tip_webapp/samples/ | `python -m pytest -q tip_webapp/tests/test_inference.py` | 180 s |
| tip_webapp API | tip_webapp/app/main.py, tip_webapp/app/inference.py, tip_webapp/__main__.py | `python -m pytest -q tip_webapp/tests/test_api.py` | 180 s |
| tip_webapp browser E2E | tip_webapp/ (server + static) | `python -m pytest -q tip_webapp/tests/test_e2e_browser.py` | 600 s |
| grayscale converter | grayscale_webapp/app/converter.py | `python -m pytest -q grayscale_webapp/tests/test_converter.py` | 60 s |
| grayscale API | grayscale_webapp/app/main.py | `python -m pytest -q grayscale_webapp/tests/test_api.py` | 60 s |
| grayscale browser E2E | grayscale_webapp/ (server + static) | `python -m pytest -q grayscale_webapp/tests/test_e2e_browser.py` | 300 s |
| val_viewer API | val_viewer/app/*.py, val_viewer/__main__.py, train_v9_tip.py, generate_synthetic_v7_mp.py (params.jsonl), v7_config.py | `python -m pytest -q val_viewer/tests/test_api.py` | 300 s |
| val_viewer browser E2E | val_viewer/ (server + static), generate_synthetic_v7_mp.py (params.jsonl), v7_config.py | `python -m pytest -q val_viewer/tests/test_e2e_browser.py` | 600 s |
| suite runner | tests/run_all.py, tests/run_all_shard.py | `python -m pytest -q tests/test_run_all.py` | 300 s |

Not parallel-safe (the runner runs each of these once per run, so one `run_all.py` is fine; two runs at once in the
same checkout, or one of these next to a `run_all.py` run, are not):
- `tests/test_studio_e2e.py` binds the fixed port 8799 (never split: its server may also build the frontend).
- `tests/test_train_v9.py::test_cli_smoke_two_epochs` uses the fixed folder `tmp_smoke/test_train_v9/`.
- `tests/test_train_gray.py` writes and removes the fixed folder `diag/m_gray/`.
- The v7 generator, look, studio and training tests start 2-4 worker pools each; running them together can exceed
  the timeouts above.
