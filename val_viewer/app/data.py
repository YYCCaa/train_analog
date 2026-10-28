"""Dataset data, model and the cached error ranking of every row of a TipNet v9 dataset.

    <data-dir>/labels.csv + PNGs --tv.build_cache--> <data-dir>_cache_images.npy (N,256,256) uint8
                                                     <data-dir>_cache_meta.npz
    tv.split_indices(N, val_split) ----> train_idx (n_train,), val_idx (n_val,) int64: dataset row of each position
    <run>/tip.pth | best_p99.pth ------> TipNet (eval) on `device`

    ranking pass (batches of `batch` consecutive rows 0..N-1, no_grad, float64 softmax)
        logits (B,1,128,128) --tv.decode_logits--> tip (B,2)   err = |tip - label|
                             --softmax, 5x5 mass around argmax--> conf (B,)
        stored in <cache_dir>/rank_<sha256 key>.npz; a warm start loads it instead.

    <data-dir>/params.jsonl (optional) --ParamStore--> saved sampled parameters of one row

An image is addressed by its dataset `row` (0..N-1, the labels.csv data-row index) everywhere. A set is
"val" (val rows), "train" (train rows) or "all"; inside a set best = ascending error, worst = descending
error, ties by row ascending.
"""

from __future__ import annotations

import hashlib
import os
import threading
import zipfile
from pathlib import Path

import numpy as np
import torch

import train_v9_tip as tv
from .params import ParamStore

# CACHE_VERSION: int, bump when the ranking maths or the cached rows change so old rank caches are ignored
CACHE_VERSION = 2
# DEFAULT_CACHE_DIR: Path, where rank_<key>.npz files live by default (gitignored)
DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[1] / ".cache"
# CONF_WIN: int, side of the window (heatmap cells) whose softmax mass is `conf`
CONF_WIN = 5
# SETS: tuple[str, ...], the image sets the viewer can rank within
SETS = ("val", "train", "all")

# Match the trainer's numerics exactly (no TF32, no autocast).
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False


def resolve_weights(run_dir) -> Path:
    """run_dir (str | Path) -> <run>/tip.pth, else <run>/best_p99.pth; else FileNotFoundError."""
    run_dir = Path(run_dir)
    for name in ("tip.pth", "best_p99.pth"):
        if (run_dir / name).is_file():
            return run_dir / name
    raise FileNotFoundError(f"no tip.pth or best_p99.pth in run directory {run_dir}")


def _file_sha(path, h) -> None:
    """Feed the bytes of the file at `path` into hashlib object `h`."""
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)


def rank_key(data_dir, val_split, img_path, weights_path, device, trainer_file=None) -> str:
    """Rank-cache key (32 hex chars) from everything the ranking depends on.

    data_dir: str | Path; val_split: float; img_path: trainer images .npy (size + mtime hashed);
    weights_path: Path; device: "cpu" | "cuda"; trainer_file: Path of train_v9_tip.py (default: the imported one)."""
    trainer_file = trainer_file or tv.__file__
    h = hashlib.sha256()
    for part in (str(data_dir), f"{val_split!r}", f"v{CACHE_VERSION}", str(device),
                 str(os.path.getsize(img_path)), str(os.stat(img_path).st_mtime_ns)):
        h.update(part.encode() + b"\0")
    _file_sha(Path(data_dir) / "labels.csv", h)
    _file_sha(weights_path, h)
    _file_sha(trainer_file, h)
    return h.hexdigest()[:32]


class ValData:
    """A v9 dataset (train + val rows), a trained TipNet and the per-image error ranking over every row.

    Attributes (N = n_total; all numpy, indexed by dataset row):
        n_total, n_train, n_val: int; names: list[str] (N,); label: (N, 2) float64 true tip (x, y) px;
        tip: (N, 2) float64 predicted tip px; err: (N,) float64 px; conf: (N,) float64 in [0, 1];
        split: (N,) str "train" | "val"; split_pos: (N,) int64 position of the row inside its split
        (train_idx[split_pos] == row for train rows, val_idx[split_pos] == row for val rows);
        train_idx: (n_train,) int64, val_idx: (n_val,) int64 dataset rows; cache_hit: bool (rank cache reused);
        weights_path: Path; run_name: str; device: str; data_dir: str; params: ParamStore.
    """

    def __init__(self, data_dir, run_dir, val_split=0.15, cache_dir=None, device="auto", batch=256):
        data_dir, run_dir = Path(data_dir), Path(run_dir)
        if not data_dir.is_dir():
            raise FileNotFoundError(f"data directory not found: {data_dir}")
        if not (data_dir / "labels.csv").is_file():
            raise FileNotFoundError(f"labels.csv not found in {data_dir}")
        if not run_dir.is_dir():
            raise FileNotFoundError(f"run directory not found: {run_dir}")
        self.weights_path: Path = resolve_weights(run_dir)  # weights file actually used
        self.run_name: str = run_dir.resolve().name         # run folder name
        self.data_dir: str = str(data_dir.resolve())        # absolute data dir
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device: str = device                           # "cpu" | "cuda"
        cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR

        # trainer cache (decoded PNGs) next to the data dir, same default prefix as train_v9_tip.py
        prefix = os.path.abspath(data_dir).rstrip("\\/") + "_cache"     # str: trainer cache prefix
        img_path, _ = tv.cache_paths(prefix)  # str: trainer images .npy
        tv.ensure_cache(self.data_dir, prefix)  # builds when missing / mismatched / labels.csv newer (trainer rule)
        self._img_path: str = img_path  # trainer images .npy, opened per read (a held memmap blocks a rebuild on Windows)
        all_names, all_labels = tv.read_labels_csv(self.data_dir)  # list[str] (N,), (N, 2) float32
        self.n_total: int = len(all_names)
        train_idx, val_idx = tv.split_indices(self.n_total, val_split)
        self.train_idx: np.ndarray = train_idx.numpy().astype(np.int64)  # (n_train,) int64 dataset rows
        self.val_idx: np.ndarray = val_idx.numpy().astype(np.int64)      # (n_val,) int64 dataset rows
        self.n_train: int = len(self.train_idx)
        self.n_val: int = len(self.val_idx)
        self.names: list[str] = all_names
        self.label: np.ndarray = all_labels.astype(np.float64)  # (N, 2)
        self.split: np.ndarray = np.full(self.n_total, "train", dtype="<U5")  # (N,) str
        self.split[self.val_idx] = "val"
        self.split_pos: np.ndarray = np.zeros(self.n_total, dtype=np.int64)   # (N,) int64
        self.split_pos[self.train_idx] = np.arange(self.n_train)
        self.split_pos[self.val_idx] = np.arange(self.n_val)
        self._rows = np.arange(self.n_total, dtype=np.int64)  # (N,) int64, 0..N-1 (what the rank cache stores as `row`)

        self._model = tv.build_model("tipnet")              # nn.Module, eval mode, on self.device
        state = torch.load(self.weights_path, map_location="cpu", weights_only=True)
        self._model.load_state_dict(state)
        self._model.to(self.device).eval()
        self._lock = threading.Lock()                       # serialises every model forward

        self.key: str = rank_key(self.data_dir, val_split, img_path, self.weights_path, self.device)
        cache_file = cache_dir / f"rank_{self.key}.npz"     # Path

        self.cache_hit: bool = False
        n = self.n_total
        if cache_file.is_file():
            try:
                with np.load(cache_file) as z:
                    tip, err, conf, row = z["tip"], z["err"], z["conf"], z["row"]
                if (np.array_equal(row, self._rows) and tip.shape == (n, 2)
                        and err.shape == (n,) and conf.shape == (n,)):
                    self.tip, self.err, self.conf = tip, err, conf  # (N,2), (N,), (N,) float64
                    self.cache_hit = True
                    print(f"[rank] cached {cache_file.name}", flush=True)
            except (OSError, ValueError, KeyError, zipfile.BadZipFile, EOFError):
                pass
        if not self.cache_hit:
            self.tip, self.err, self.conf = self._rank(batch)
            cache_dir.mkdir(parents=True, exist_ok=True)
            tmp = cache_file.with_suffix(".partial.npz")
            np.savez(tmp, tip=self.tip, err=self.err, conf=self.conf, row=self._rows, label=self.label)
            os.replace(tmp, cache_file)

        self.params: ParamStore = ParamStore(self.data_dir, self.names, cache_dir, self.label)  # saved parameters, if any
        # dict[str, (n_set,) int64]: dataset rows of each set, ascending
        self._set_rows = {"val": np.sort(self.val_idx), "train": np.sort(self.train_idx), "all": self._rows}
        # dict[(set, side), (n_set,) int64]: rows of the set ordered best = ascending (err, row),
        # worst = descending err, ascending row
        self._order = {}
        for name, rows in self._set_rows.items():
            e = self.err[rows]
            self._order[name, "best"] = rows[np.lexsort((rows, e))]
            self._order[name, "worst"] = rows[np.lexsort((rows, -e))]

    def _rank(self, batch):
        """Forward every row -> (tip (N,2), err (N,), conf (N,)) float64."""
        n = self.n_total
        tip = np.zeros((n, 2))   # float64 predicted tips
        conf = np.zeros(n)       # float64 5x5 softmax mass
        r = CONF_WIN // 2
        for s in range(0, n, batch):
            mm = np.load(self._img_path, mmap_mode="r")     # (N,256,256) uint8 memmap
            x = torch.from_numpy(np.array(mm[s:s + batch])).to(self.device)  # (B,256,256) uint8
            del mm
            with self._lock, torch.no_grad():
                logits = self._model(tv.to_input(x))        # (B,1,128,128) float32
                tip[s:s + len(x)] = tv.decode_logits(logits).double().cpu().numpy()
                lg = logits[:, 0].double()                  # (B,128,128) float64
                p = torch.softmax(lg.flatten(1), dim=1).reshape(lg.shape).cpu().numpy()
                arg = logits[:, 0].flatten(1).argmax(dim=1).cpu().numpy()  # (B,) int64 flat argmax
            h, w = p.shape[1:]
            for b, a in enumerate(arg):
                iy, ix = divmod(int(a), w)
                conf[s + b] = p[b, max(iy - r, 0):iy + r + 1, max(ix - r, 0):ix + r + 1].sum()
            print(f"[rank] {min(s + batch, n)}/{n}", flush=True)
        err = np.hypot(*(tip - self.label).T)               # (N,) float64
        return tip, err, conf

    def image(self, row: int) -> np.ndarray:
        """Dataset row -> (256, 256) uint8 dial image."""
        mm = np.load(self._img_path, mmap_mode="r")  # (N,256,256) uint8 memmap, closed on return
        return np.array(mm[row])

    def logits(self, row: int) -> np.ndarray:
        """Dataset row -> (128, 128) float64 logits of one forward pass (thread-safe)."""
        x = torch.from_numpy(self.image(row)).unsqueeze(0).to(self.device)  # (1,256,256) uint8
        with self._lock, torch.no_grad():
            out = self._model(tv.to_input(x))               # (1,1,128,128) float32
        return out[0, 0].double().cpu().numpy()

    def entries(self, set_name: str, side: str, n: int) -> list[dict]:
        """set_name 'val' | 'train' | 'all', side 'best' | 'worst', n (int >= 1) -> up to n entry dicts
        {rank (1-based), row, split ('train' | 'val'), split_pos, name, label [x, y], tip [x, y], err, conf}."""
        order = self._order[set_name, side][:n]
        return [{"rank": i + 1, "row": int(r), "split": str(self.split[r]), "split_pos": int(self.split_pos[r]),
                 "name": self.names[r], "label": [float(v) for v in self.label[r]],
                 "tip": [float(v) for v in self.tip[r]], "err": float(self.err[r]), "conf": float(self.conf[r])}
                for i, r in enumerate(order)]

    def summary(self, set_name: str = "val") -> dict:
        """Header numbers: run, weights file name, data_dir, set, n (images in the set), n_train, n_val, n_total,
        p99, max_err (float | None when the set is empty), n_over_3, n_over_7 (of that set), device, key (rank-cache key, str),
        params ('saved' | 'none')."""
        err = self.err[self._set_rows[set_name]]  # (n_set,) float64
        empty = len(err) == 0  # bool: a set with no image has no percentile / maximum
        return {"run": self.run_name, "weights": self.weights_path.name, "data_dir": self.data_dir,
                "set": set_name, "n": len(err), "n_train": self.n_train, "n_val": self.n_val,
                "n_total": self.n_total, "p99": None if empty else float(np.percentile(err, 99)),
                "max_err": None if empty else float(err.max()),
                "n_over_3": int((err > 3).sum()), "n_over_7": int((err > 7).sum()), "device": self.device,
                "key": self.key, "params": "saved" if self.params.available else "none"}
