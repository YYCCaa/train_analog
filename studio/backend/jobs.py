"""Single generation job runner: generate_dataset in a background thread."""
import threading

import generate_synthetic_v7_mp as gen


class JobRunner:
    """Runs at most one dataset generation at a time; thread-safe status."""

    def __init__(self):
        self._lock = threading.Lock()  # guards every field below
        self._thread = None  # worker thread or None (threading.Thread)
        self._cancel = threading.Event()  # set to stop the running job
        self._state = "idle"  # idle|running|done|cancelled|error (str)
        self._done = 0  # finished images (int)
        self._total = 0  # requested images (int)
        self._output_dir = None  # str|None
        self._error = None  # str|None

    def start(self, cfg, output_dir, num_images, seed, workers, overwrite, save_params=False):
        """Start a job. Raises RuntimeError if one is running, OutputDirNotEmpty
        (synchronously) if the folder holds old data and overwrite is False.
        save_params (bool): also write params.jsonl."""
        with self._lock:
            if self._state == "running":
                raise RuntimeError("a generation job is already running")
            gen.check_output_dir(output_dir, overwrite)
            self._cancel = cancel = threading.Event()
            self._state, self._done, self._total = "running", 0, int(num_images)
            self._output_dir, self._error = output_dir, None
            thread = threading.Thread(
                target=self._run, daemon=True,
                args=(cfg, output_dir, num_images, seed, workers, overwrite, cancel,
                      save_params))
            try:
                thread.start()
            except BaseException:
                self._state = "idle"  # never leave a phantom running job
                raise
            self._thread = thread

    def _progress(self, done, total):
        """progress_cb: record done/total (ints) under the lock."""
        with self._lock:
            self._done, self._total = done, total

    def _run(self, cfg, output_dir, num_images, seed, workers, overwrite, cancel,
             save_params=False):
        """Thread body: run generate_dataset, then publish the final state."""
        try:
            res = gen.generate_dataset(
                cfg, output_dir, num_images, seed=seed, workers=workers,
                overwrite=overwrite, progress_cb=self._progress, cancel_event=cancel,
                save_params=save_params)
            state, err = ("done" if res["completed"] else "cancelled"), None
        except BaseException as e:  # report any failure to the UI
            state, err = "error", f"{type(e).__name__}: {e}"
        with self._lock:
            self._state, self._error = state, err

    def status(self):
        """-> dict state/done/total/output_dir/error."""
        with self._lock:
            return {"state": self._state, "done": self._done, "total": self._total,
                    "output_dir": self._output_dir, "error": self._error}

    def cancel(self):
        """Request cancel of the running job; returns the current status."""
        self._cancel.set()
        return self.status()

    def join(self, timeout=None):
        """Wait for the job thread (timeout: float seconds|None)."""
        t = self._thread
        if t is not None:
            t.join(timeout)
