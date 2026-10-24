"""Bake a shape bank of organic object silhouettes from Places365 with SAM.

Offline, one-time tool. SAM ViT-B (Segment Anything) segments a seeded random
sample of Places365 photos; mid-sized, mostly-interior masks are normalised
into 128x128 bit frames and written to one .npz that the v7 generator samples
for reflection/glare shapes (override path with V7_SHAPE_BANK).

    photo (H,W,3) --SAM auto masks--> records --filter_masks--> <= max_per_photo
        --normalise_mask--> (128,128) bool frame + (h,w) --pack_bank--> .npz

File format (binding contract, see load_bank for validation):
    masks uint8 (N, 2048)  packbits of each bool (128,128) frame
    hw    uint16 (N, 2)    silhouette (h, w), anchored top-left, max(h,w)==128
    src   int32 (N,)       index of the source photo in the sorted photo list
    meta  0-d str (json)   version/frame/model/n_photos/places_dir/created

Usage:
    python tools/build_shape_bank.py                       # 5000 photos -> ~/places365_shapes.npz
    python tools/build_shape_bank.py --n-photos 20 --out tmp_smoke/bank.npz
    python tools/build_shape_bank.py --resume              # continue a stopped run
"""

import argparse
import datetime as _dt
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

FRAME = 128  # side of the square silhouette frame, px (int)
PACKED = FRAME * FRAME // 8  # bytes per packed frame row (int, 2048)
BANK_VERSION = 1  # contract version written to meta (int)
MODEL_NAME = "sam_vit_b"  # model tag written to meta (str)
AREA_MIN, AREA_MAX = 0.003, 0.35  # kept mask area as fraction of the photo (float, float)
MAX_BORDERS = 2  # masks touching more image borders than this are dropped (int)
IMG_EXTS = {".jpg", ".jpeg", ".png"}  # photo file extensions scanned (set[str])
SAM_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"  # weights URL (str)
SAM_CKPT = Path.home() / ".cache" / "sam" / "sam_vit_b_01ec64.pth"  # local weights path (Path)
# SAM automatic mask generator settings (dict[str, number]); points_per_side=16 is the speed choice
SAM_KW = dict(points_per_side=16, pred_iou_thresh=0.86,
              stability_score_thresh=0.92, min_mask_region_area=100)
REQUIRED_META = ("version", "frame", "model", "n_photos", "places_dir", "created")  # meta keys (tuple[str])


# ====================================================================== pure helpers

def _borders_touched(seg):
    """Number of image borders (0..4) that bool (H, W) mask `seg` touches (int)."""
    return int(seg[0, :].any()) + int(seg[-1, :].any()) + int(seg[:, 0].any()) + int(seg[:, -1].any())


def filter_masks(records, img_hw, max_per_photo):
    """Keep mid-sized, mostly-interior masks, best first.

    records: list of SAM record dicts, each with "segmentation" bool (H, W),
             "predicted_iou" float and "stability_score" float.
    img_hw: (H, W) photo size (tuple[int, int]).
    max_per_photo: top-k cap (int).
    Returns a list (<= max_per_photo) of the kept records, sorted by
    predicted_iou * stability_score descending.
    """
    total = float(img_hw[0] * img_hw[1])  # photo pixel count (float)
    kept = []  # surviving records (list[dict])
    for r in records:
        seg = np.asarray(r["segmentation"], bool)  # (H, W) bool
        frac = seg.sum() / total  # area fraction (float)
        if not (AREA_MIN <= frac <= AREA_MAX):
            continue
        if _borders_touched(seg) > MAX_BORDERS:
            continue
        kept.append(r)
    kept.sort(key=lambda r: float(r["predicted_iou"]) * float(r["stability_score"]), reverse=True)
    return kept[:max_per_photo]


def normalise_mask(bool_mask):
    """Crop to tight bbox, NEAREST-resize longest side to FRAME, place top-left.

    bool_mask: (H, W) bool, must be non-empty.
    Returns (frame, h, w): frame (FRAME, FRAME) bool, silhouette in frame[:h, :w];
    h, w ints with max(h, w) == FRAME and 1 <= h, w.
    """
    m = np.asarray(bool_mask, bool)  # (H, W) bool
    ys, xs = np.nonzero(m)  # foreground coords, each (K,) int64
    if ys.size == 0:
        raise ValueError("normalise_mask: empty mask")
    crop = m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]  # (bh, bw) bool tight bbox
    bh, bw = crop.shape
    s = FRAME / max(bh, bw)  # scale factor (float)
    h = FRAME if bh >= bw else max(1, min(FRAME, int(round(bh * s))))  # target rows (int)
    w = FRAME if bw > bh else max(1, min(FRAME, int(round(bw * s))))  # target cols (int)
    small = np.asarray(Image.fromarray(crop.astype(np.uint8) * 255).resize((w, h), Image.NEAREST)) > 0
    frame = np.zeros((FRAME, FRAME), bool)  # (FRAME, FRAME) bool output
    frame[:h, :w] = small
    return frame, h, w


def pack_bank(frames, hws, srcs, meta):
    """Pack frames into the contract dict.

    frames: list of (FRAME, FRAME) bool; hws: list of (h, w); srcs: list of int
    photo indices; meta: dict with at least n_photos and places_dir (other
    contract fields are filled in when missing).
    Returns dict: masks uint8 (N, 2048), hw uint16 (N, 2), src int32 (N,),
    meta 0-d str array (json).
    """
    n = len(frames)  # number of silhouettes (int)
    masks = np.zeros((n, PACKED), np.uint8)  # (N, 2048) uint8
    for k, f in enumerate(frames):
        masks[k] = np.packbits(np.asarray(f, bool).reshape(-1))
    full = {"version": BANK_VERSION, "frame": FRAME, "model": MODEL_NAME,
            "created": _dt.datetime.now().isoformat(timespec="seconds")}  # meta defaults (dict)
    full.update(meta)
    return {"masks": masks,
            "hw": np.asarray(hws, np.uint16).reshape(n, 2),
            "src": np.asarray(srcs, np.int32).reshape(n),
            "meta": np.array(json.dumps(full))}


def save_bank(path, bank):
    """Atomically write the bank dict to `path` with np.savez_compressed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")  # temp file next to target (Path)
    with open(tmp, "wb") as fh:  # file handle stops numpy appending ".npz"
        np.savez_compressed(fh, **bank)
    os.replace(tmp, path)


def load_bank(path):
    """Load and validate a bank .npz against the contract.

    Returns dict with masks uint8 (N, 2048), hw uint16 (N, 2), src int32 (N,),
    meta 0-d str array. Raises ValueError naming the offending key.
    """
    with np.load(path, allow_pickle=False) as z:
        missing = [k for k in ("masks", "hw", "src", "meta") if k not in z.files]
        if missing:
            raise ValueError(f"shape bank {path}: missing key(s) {missing}")
        bank = {k: z[k] for k in ("masks", "hw", "src", "meta")}
    masks, hw, src, meta = bank["masks"], bank["hw"], bank["src"], bank["meta"]
    if masks.dtype != np.uint8 or masks.ndim != 2 or masks.shape[1] != PACKED:
        raise ValueError(f"shape bank: masks must be uint8 (N, {PACKED}), got {masks.dtype} {masks.shape}")
    n = masks.shape[0]  # number of silhouettes (int)
    if hw.dtype != np.uint16 or hw.shape != (n, 2):
        raise ValueError(f"shape bank: hw must be uint16 ({n}, 2), got {hw.dtype} {hw.shape}")
    if src.dtype != np.int32 or src.shape != (n,):
        raise ValueError(f"shape bank: src must be int32 ({n},), got {src.dtype} {src.shape}")
    if n and (hw.min() < 1 or hw.max() > FRAME or not np.all(hw.max(axis=1) == FRAME)):
        raise ValueError(f"shape bank: hw values must be in [1, {FRAME}] with max(h, w) == {FRAME}")
    if meta.shape != () or meta.dtype.kind != "U":
        raise ValueError(f"shape bank: meta must be a 0-d str array, got {meta.dtype} {meta.shape}")
    try:
        m = json.loads(str(meta))
    except json.JSONDecodeError as e:
        raise ValueError(f"shape bank: meta is not json ({e})") from None
    lack = [k for k in REQUIRED_META if k not in m]
    if lack:
        raise ValueError(f"shape bank: meta missing {lack}")
    return bank


# ====================================================================== SAM / IO

def list_photos(places):
    """Sorted list of photo Paths under `places` (recursive, jpg/jpeg/png)."""
    return sorted(p for p in Path(places).rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS)


def ensure_checkpoint(path=SAM_CKPT):
    """Return the SAM ViT-B weights path, downloading it with progress if missing."""
    path = Path(path)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")  # partial download (Path)
    print(f"downloading SAM weights to {path} ...", flush=True)
    last = [0]  # last printed percent, boxed for the closure (list[int])

    def hook(blocks, bsize, total):
        pct = int(100 * blocks * bsize / total) if total > 0 else 0
        if pct >= last[0] + 5:
            last[0] = pct
            print(f"  {min(pct, 100)}%", flush=True)

    urllib.request.urlretrieve(SAM_URL, tmp, hook)
    os.replace(tmp, path)
    return path


def make_sam_generator(device):
    """Build the real SamAutomaticMaskGenerator; returns callable rgb(H,W,3) uint8 -> records."""
    import torch
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
    sam = sam_model_registry["vit_b"](checkpoint=str(ensure_checkpoint()))
    sam.to(device).eval()
    gen = SamAutomaticMaskGenerator(sam, **SAM_KW)

    def run(rgb):
        with torch.inference_mode():
            return gen.generate(rgb)
    return run


def make_fake_generator():
    """Deterministic SAM stand-in for tests: a few ellipses keyed on photo content."""
    def run(rgb):
        h, w = rgb.shape[:2]
        rng = np.random.default_rng(int(rgb.astype(np.int64).sum()) % (2 ** 32))
        yy, xx = np.mgrid[0:h, 0:w]
        recs = []  # SAM-style record dicts (list[dict])
        for _ in range(4):
            cy, cx = rng.uniform(0.3, 0.7) * h, rng.uniform(0.3, 0.7) * w
            ry, rx = rng.uniform(0.05, 0.25) * h, rng.uniform(0.05, 0.25) * w
            seg = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1.0
            recs.append({"segmentation": seg, "area": int(seg.sum()),
                         "predicted_iou": float(rng.uniform(0.86, 1.0)),
                         "stability_score": float(rng.uniform(0.92, 1.0))})
        return recs
    return run


def _side_paths(out):
    """(partial npz, progress json) paths beside `out` (tuple[Path, Path])."""
    out = Path(out)
    stem = out.name[:-4] if out.name.endswith(".npz") else out.name
    return out.with_name(stem + ".partial.npz"), out.with_name(stem + ".progress.json")


def _unpack(bank):
    """Bank dict -> (frames list[(FRAME,FRAME) bool], hws list[(h,w)], srcs list[int])."""
    frames = [np.unpackbits(r)[:FRAME * FRAME].reshape(FRAME, FRAME).astype(bool) for r in bank["masks"]]
    return frames, [tuple(int(v) for v in x) for x in bank["hw"]], [int(s) for s in bank["src"]]


def bake(args, generator):
    """Run the bake loop with `generator` (rgb -> records). Returns exit code (int)."""
    places = Path(args.places).expanduser()
    out = Path(args.out).expanduser()
    partial, progress = _side_paths(out)
    photos = list_photos(places)  # sorted photo list; src indexes into it (list[Path])
    if not photos:
        print(f"no photos found under {places}", file=sys.stderr)
        return 2
    n = min(args.n_photos, len(photos))  # photos to process (int)
    rng = np.random.default_rng(args.seed)
    chosen = sorted(rng.choice(len(photos), size=n, replace=False).tolist())  # photo indices (list[int])
    run_key = {"places_dir": str(places), "n_photos": n, "seed": args.seed,
               "max_per_photo": args.max_per_photo, "max_masks": args.max_masks}  # identifies a run (dict)

    frames, hws, srcs = [], [], []  # accumulated silhouettes (lists)
    start = 0  # position in `chosen` to process next (int)
    if args.resume and partial.exists() and progress.exists():
        prog = json.loads(progress.read_text())
        if prog.get("run") != run_key:
            print(f"--resume: progress {progress} is for a different run {prog.get('run')}", file=sys.stderr)
            return 2
        frames, hws, srcs = _unpack(load_bank(partial))
        start = int(prog["next"])
        print(f"resuming at photo {start}/{n} with {len(frames)} masks", flush=True)
    elif args.resume:
        print("--resume: no checkpoint found, starting fresh", flush=True)

    def checkpoint(next_pos):
        save_bank(partial, pack_bank(frames, hws, srcs, {"n_photos": next_pos, "places_dir": str(places)}))
        tmp = progress.with_name(progress.name + ".tmp")
        tmp.write_text(json.dumps({"run": run_key, "next": next_pos, "n_masks": len(frames)}))
        os.replace(tmp, progress)

    t0 = time.time()
    done = start  # photos processed so far (int)
    for pos in range(start, n):
        if len(frames) >= args.max_masks:
            break
        if args.stop_after is not None and pos - start >= args.stop_after:
            print(f"stop-after {args.stop_after}: exiting without finishing", flush=True)
            return 3
        idx = chosen[pos]  # index into the sorted photo list (int)
        try:
            rgb = np.asarray(Image.open(photos[idx]).convert("RGB"))  # (H, W, 3) uint8
        except OSError as e:
            print(f"skip unreadable {photos[idx]}: {e}", file=sys.stderr)
            done = pos + 1
            continue
        for r in filter_masks(generator(rgb), rgb.shape[:2], args.max_per_photo):
            if len(frames) >= args.max_masks:
                break
            f, h, w = normalise_mask(r["segmentation"])
            frames.append(f); hws.append((h, w)); srcs.append(idx)
        done = pos + 1
        if done % args.checkpoint_every == 0 and done < n:
            checkpoint(done)
        if done % 25 == 0 or done == n:
            el = time.time() - t0
            rate = el / max(done - start, 1)
            print(f"[{done}/{n}] masks={len(frames)} {rate:.2f}s/photo eta={rate * (n - done) / 60:.1f}min",
                  flush=True)

    save_bank(out, pack_bank(frames, hws, srcs, {"n_photos": done, "places_dir": str(places)}))
    for p in (partial, progress):
        if p.exists():
            p.unlink()
    el = time.time() - t0
    print(f"wrote {out}: {len(frames)} masks from {done} photos in {el:.1f}s "
          f"({el / max(done - start, 1):.2f}s/photo)", flush=True)
    return 0


def parse_args(argv=None):
    """CLI arguments (argparse.Namespace)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--places", default=str(Path.home() / "places365"), help="photo folder, scanned recursively")
    ap.add_argument("--out", default=str(Path.home() / "places365_shapes.npz"), help="output .npz")
    ap.add_argument("--n-photos", type=int, default=5000, help="seeded random sample size")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None, help="torch device (default cuda if available)")
    ap.add_argument("--max-per-photo", type=int, default=10)
    ap.add_argument("--max-masks", type=int, default=30000)
    ap.add_argument("--checkpoint-every", type=int, default=250, help="photos between partial checkpoints")
    ap.add_argument("--resume", action="store_true", help="continue from <out stem>.partial.npz")
    ap.add_argument("--fake-generator", action="store_true", help=argparse.SUPPRESS)  # tests: no SAM
    ap.add_argument("--stop-after", type=int, default=None, help=argparse.SUPPRESS)  # tests: simulate a crash
    return ap.parse_args(argv)


def main(argv=None, generator=None):
    """Entry point; `generator` (rgb -> SAM records) is an injection seam for tests."""
    args = parse_args(argv)
    if generator is None:
        if args.fake_generator:
            generator = make_fake_generator()
        else:
            import torch
            device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
            print(f"loading SAM vit_b on {device}", flush=True)
            generator = make_sam_generator(device)
    return bake(args, generator)


if __name__ == "__main__":
    sys.exit(main())
