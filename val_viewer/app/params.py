"""Read single records of a dataset's `params.jsonl` without loading the file.

    <data-dir>/params.jsonl   line i = record of labels.csv data row i:
        {"filename", "seed", "angle_deg", "tip_x", "tip_y", "sampled"}   (strict JSON, one object per line)
    first use --scan for b"\\n"--> bounds (n_lines + 1,) int64 byte offsets: line i = bytes bounds[i]:bounds[i+1]
                                   cached in <cache_dir>/params_<sha256 key>.npy (key: path, size, mtime)
    get(row) --stat, seek + read one line--> dict, or NoParamsFile / NoRecord / RecordMismatch / ParamsChanged

Non-finite floats arrive as the strings "NaN" / "Infinity" / "-Infinity" and stay strings; a bare NaN token or an
overflowing number such as 1e999 (a hand-edited file) is turned into the same string so the HTTP answer stays
strict JSON. A leading UTF-8 BOM is ignored.
A record is refused when its file name or its tip differs from the row's labels.csv entry (a params.jsonl from
another run of the generator has the same file names), and every get() notices that the file changed or vanished.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path

import numpy as np


# TIP_TOL: float, largest |record tip - labels.csv tip| in px that still counts as the same image
# (labels.csv holds 3 decimals and is read back as float32)
TIP_TOL = 1e-3


class NoParamsFile(Exception):
    """The dataset folder holds no params.jsonl (or it was deleted while the viewer runs)."""


class ParamsChanged(Exception):
    """params.jsonl has another size or mtime than the one the line index was built from."""


class NoRecord(Exception):
    """The file has no complete record for this row (missing, blank or torn line)."""


class RecordMismatch(Exception):
    """The record's file name or tip differs from the row's labels.csv entry.

    Attribute record_name: str, the file name found in the record."""

    def __init__(self, record_name: str):
        super().__init__(record_name)
        self.record_name = record_name


def _parse_float(text: str):
    """json parse_float hook: the float of a number token, or "Infinity" / "-Infinity" if it overflows."""
    v = float(text)  # float
    if v in (float("inf"), float("-inf")):
        return "Infinity" if v > 0 else "-Infinity"
    return v


class ParamStore:
    """Random access to the records of <data_dir>/params.jsonl.

    Attributes: available: bool, the file exists; path: Path of the file;
    names: list[str] (N,), labels.csv file name of each dataset row;
    labels: (N, 2) float64 | None, labels.csv tip (x, y) px of each row (None: the tip is not checked)."""

    def __init__(self, data_dir, names: list[str], cache_dir, labels: np.ndarray | None = None):
        self.path: Path = Path(data_dir) / "params.jsonl"
        self.available: bool = self.path.is_file()
        self.names: list[str] = names
        self.labels: np.ndarray | None = labels
        self._sig: tuple[int, int] | None = None  # (size bytes, mtime_ns) of the file the index was built from
        self._cache_dir: Path = Path(cache_dir)
        self._bounds: np.ndarray | None = None  # (n_lines + 1,) int64 line byte offsets, built on first get
        self._lock = threading.Lock()           # guards the one-time index build

    def _index(self) -> np.ndarray:
        """Line byte offsets (n_lines + 1,) int64; read from the cache file or scanned and cached."""
        with self._lock:
            if self._bounds is not None:
                return self._bounds
            try:
                st = os.stat(self.path)
            except FileNotFoundError:
                raise NoParamsFile() from None
            self._sig = (st.st_size, st.st_mtime_ns)
            key = hashlib.sha256(f"{self.path.resolve()}\0{st.st_size}\0{st.st_mtime_ns}".encode()).hexdigest()[:32]
            cache_file = self._cache_dir / f"params_{key}.npy"  # Path of the cached offsets
            bounds = None
            try:
                bounds = np.load(cache_file)
                if bounds.ndim != 1 or len(bounds) < 1 or bounds[-1] != st.st_size:
                    bounds = None
            except (OSError, ValueError):
                bounds = None
            if bounds is None:
                starts = [0]  # list[int]: byte offset where each line starts
                pos = 0       # int: bytes consumed so far
                with self._open() as f:
                    for chunk in iter(lambda: f.read(1 << 20), b""):
                        nl = np.flatnonzero(np.frombuffer(chunk, dtype=np.uint8) == 10)
                        starts.extend((pos + nl + 1).tolist())
                        pos += len(chunk)
                if starts[-1] >= pos:  # the file ends with a newline (or is empty): no line starts there
                    starts.pop()
                bounds = np.asarray(starts + [pos], dtype=np.int64)
                try:
                    self._cache_dir.mkdir(parents=True, exist_ok=True)
                    tmp = cache_file.with_suffix(".partial.npy")
                    np.save(tmp, bounds)
                    os.replace(tmp, cache_file)
                except OSError:
                    pass  # the cache only saves a rescan
            self._bounds = bounds
            return bounds

    def _open(self):
        """Open the file for binary reading; a vanished file raises NoParamsFile."""
        try:
            return open(self.path, "rb")
        except FileNotFoundError:
            raise NoParamsFile() from None

    def get(self, row: int) -> dict:
        """Dataset row (int) -> its record dict (float tokens as parsed; non-finite ones as strings).

        Raises NoParamsFile (no file, or deleted since the start), ParamsChanged (size or mtime differ from the
        indexed file), NoRecord, RecordMismatch (other file name, or tip off by more than TIP_TOL px)."""
        if not self.available:
            raise NoParamsFile()
        try:
            st = os.stat(self.path)
        except FileNotFoundError:
            raise NoParamsFile() from None
        bounds = self._index()
        if self._sig != (st.st_size, st.st_mtime_ns):
            raise ParamsChanged()
        if not 0 <= row < len(bounds) - 1:
            raise NoRecord()
        with self._open() as f:
            f.seek(int(bounds[row]))
            raw = f.read(int(bounds[row + 1] - bounds[row])).strip()  # strip drops \n, \r\n and blanks
        if not raw:
            raise NoRecord()
        try:
            rec = json.loads(raw.decode("utf-8-sig"), parse_constant=lambda s: s, parse_float=_parse_float)
        except ValueError:  # JSONDecodeError and UnicodeDecodeError
            raise NoRecord() from None
        if not isinstance(rec, dict):
            raise NoRecord()
        if rec.get("filename") != self.names[row]:
            raise RecordMismatch(str(rec.get("filename")))
        if self.labels is not None:
            for col, key in enumerate(("tip_x", "tip_y")):
                v = rec.get(key)  # int | float | str | None, the record's tip coordinate
                if (not isinstance(v, (int, float)) or isinstance(v, bool)
                        or not abs(v - self.labels[row][col]) <= TIP_TOL):
                    raise RecordMismatch(str(rec.get("filename")))
        return rec
