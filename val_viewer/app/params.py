"""Read single records of a dataset's `params.jsonl` without loading the file.

    <data-dir>/params.jsonl   line i = record of labels.csv data row i:
        {"filename", "seed", "angle_deg", "tip_x", "tip_y", "sampled"}   (strict JSON, one object per line)
    first use --scan for b"\\n"--> bounds (n_lines + 1,) int64 byte offsets: line i = bytes bounds[i]:bounds[i+1]
                                   cached in <cache_dir>/params_<sha256 key>.npy (key: path, size, mtime)
    get(row) --seek + read one line--> dict, or NoParamsFile / NoRecord / RecordMismatch

Non-finite floats arrive as the strings "NaN" / "Infinity" / "-Infinity" and stay strings; a bare NaN token
(a hand-edited file) is turned into the same string so the HTTP answer stays strict JSON.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path

import numpy as np


class NoParamsFile(Exception):
    """The dataset folder holds no params.jsonl."""


class NoRecord(Exception):
    """The file has no complete record for this row (missing, blank or torn line)."""


class RecordMismatch(Exception):
    """The record's file name differs from the row's labels.csv name.

    Attribute record_name: str, the file name found in the record."""

    def __init__(self, record_name: str):
        super().__init__(record_name)
        self.record_name = record_name


class ParamStore:
    """Random access to the records of <data_dir>/params.jsonl.

    Attributes: available: bool, the file exists; path: Path of the file;
    names: list[str] (N,), labels.csv file name of each dataset row."""

    def __init__(self, data_dir, names: list[str], cache_dir):
        self.path: Path = Path(data_dir) / "params.jsonl"
        self.available: bool = self.path.is_file()
        self.names: list[str] = names
        self._cache_dir: Path = Path(cache_dir)
        self._bounds: np.ndarray | None = None  # (n_lines + 1,) int64 line byte offsets, built on first get
        self._lock = threading.Lock()           # guards the one-time index build

    def _index(self) -> np.ndarray:
        """Line byte offsets (n_lines + 1,) int64; read from the cache file or scanned and cached."""
        with self._lock:
            if self._bounds is not None:
                return self._bounds
            st = os.stat(self.path)
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
                with open(self.path, "rb") as f:
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

    def get(self, row: int) -> dict:
        """Dataset row (int) -> its record dict; NoParamsFile, NoRecord or RecordMismatch otherwise."""
        if not self.available:
            raise NoParamsFile()
        bounds = self._index()
        if not 0 <= row < len(bounds) - 1:
            raise NoRecord()
        with open(self.path, "rb") as f:
            f.seek(int(bounds[row]))
            raw = f.read(int(bounds[row + 1] - bounds[row])).strip()  # strip drops \n, \r\n and blanks
        if not raw:
            raise NoRecord()
        try:
            rec = json.loads(raw.decode("utf-8"), parse_constant=lambda s: s)
        except ValueError:  # JSONDecodeError and UnicodeDecodeError
            raise NoRecord() from None
        if not isinstance(rec, dict):
            raise NoRecord()
        if rec.get("filename") != self.names[row]:
            raise RecordMismatch(str(rec.get("filename")))
        return rec
