# train_dial

Analog-dial autocalibration: synthetic dial generators (v3..v7), tip / ellipse models, TFLite export, and three
local web apps (`tip_webapp/`, `grayscale_webapp/`, `studio/`). Data and training runs live outside the repo in
`../train_dial_data` (`sv9_150k`, `runs/`).

## Tests

Logs: `tmp_smoke/testlogs/<task>/<timestamp>/` (gitignored), written by
`python C:/Users/Administrator/.claude/hooks/wt.py testlog <task> <label> -- <command>`.

Run every command from the repo root. No test needs a GPU, the network, or the real dataset; datasets are synthetic
and written to tmp_path / tempfile. Timeouts are in seconds.

| area | source files | test command | timeout |
|---|---|---|---|
| v7 config parse/validate | v7_config.py, tests/data/*.yaml | `python -m pytest -q tests/test_v7_config.py` | 60 |
| v7 renderer | generate_synthetic_v7_mp.py, v7_config.py | `python -m pytest -q tests/test_v7_render.py` | 600 |
| v7 generator CLI | generate_synthetic_v7_mp.py, v7_config.py | `python -m pytest -q tests/test_v7_gray_generator.py` | 600 |
| v7 look-match (slow) | tools/compare_v7_look.py, generate_synthetic_v7_mp.py | `python -m pytest -q tests/test_v7_look.py` | 1200 |
| v4 generator | generate_synthetic_v4.py, generate_synthetic_v3.py | `python -m pytest -q tests/test_v4_generator.py` | 60 |
| autocal math | autocal_utils.py | `python -m pytest -q tests/test_autocal_math.py` | 120 |
| autocal models | autocal_utils.py | `python -m pytest -q tests/test_autocal_models.py` | 120 |
| autocal dataset | autocal_utils.py | `python -m pytest -q tests/test_autocal_dataset.py` | 120 |
| v9 TipNet training | train_v9_tip.py, generate_synthetic_v7_mp.py, v7_config.py | `python -m pytest -q tests/test_train_v9.py` | 1200 |
| v9 TFLite export | export_v9_tflite.py, train_v9_tip.py | `python -m pytest -q tests/test_export_v9_tflite.py` | 1200 |
| gray InvResNet training | train_e_inv_res_gray.py, train_utils_v2_gray.py | `python -m pytest -q tests/test_train_gray.py` | 900 |
| studio API | studio/backend/*.py, generate_synthetic_v7_mp.py | `python -m pytest -q tests/test_studio_api.py` | 600 |
| studio browser E2E | studio/run_studio.py, studio/backend/*, studio/frontend | `python -m pytest -q tests/test_studio_e2e.py` | 1200 |
| tip_webapp inference | tip_webapp/app/inference.py, tip_webapp/model/, tip_webapp/samples/ | `python -m pytest -q tip_webapp/tests/test_inference.py` | 180 |
| tip_webapp API | tip_webapp/app/main.py, tip_webapp/app/inference.py, tip_webapp/__main__.py | `python -m pytest -q tip_webapp/tests/test_api.py` | 180 |
| tip_webapp browser E2E | tip_webapp/ (server + static) | `python -m pytest -q tip_webapp/tests/test_e2e_browser.py` | 600 |
| grayscale converter | grayscale_webapp/app/converter.py | `python -m pytest -q grayscale_webapp/tests/test_converter.py` | 60 |
| grayscale API | grayscale_webapp/app/main.py | `python -m pytest -q grayscale_webapp/tests/test_api.py` | 60 |
| grayscale browser E2E | grayscale_webapp/ (server + static) | `python -m pytest -q grayscale_webapp/tests/test_e2e_browser.py` | 300 |
| val_viewer API | val_viewer/app/*.py, val_viewer/__main__.py, train_v9_tip.py | `python -m pytest -q val_viewer/tests/test_api.py` | 300 |
| val_viewer browser E2E | val_viewer/ (server + static) | `python -m pytest -q val_viewer/tests/test_e2e_browser.py` | 600 |

Not parallel-safe:
- `tests/test_studio_e2e.py` binds the fixed port 8799.
- `tests/test_train_v9.py::test_cli_smoke_two_epochs` uses the fixed folder `tmp_smoke/test_train_v9/`.
- `tests/test_train_gray.py` writes and removes the fixed folder `diag/m_gray/`.
- The v7 generator, look, studio and training tests start 2-4 worker pools each; running them together can exceed
  the timeouts above.
