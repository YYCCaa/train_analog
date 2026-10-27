"""Step 2: small learned models on the cheap map features (features.py) -> P(err > t px) for a few t.

    feats (n, 41) float32 ─standardise─▶ LR (41 -> K)  or  MLP (41 -> 32 -> 32 -> K), K = len(THRESH)
                                          │
                                          ▼  score = -logit P(err > 3 px)   (higher = more trustworthy)

Fitting data (README rule 2; val is never used to choose anything):
    train clean (20000) + trainnoisy (K_NOISY x 20000 noisy copies, their own noisy errors as labels)
    hyper-parameters: 5-fold CV on the train sample, grouped by source image (common.TRAIN_FOLD),
                      judged on the clean held-out train images' auroc_err3 / ap_err3
    final protocols:
      "trainfit"  fit on all train + noisy train, score val; train rows are the 5-fold fold-out scores
      "xfit"      fit on train + noisy train + val fold a, score val fold 1-a (both ways); train rows are
                  IN-SAMPLE (mean of the two fold models' scores, train is in both fits)

Run from the worktree root:  python research/confidence/learned/fit_feats.py cv | final
"""

import sys
import time

import numpy as np
import torch
import torch.nn as nn

import common as C
import features as ft
import eval_conf as ev

# THRESH: (K,) float, error thresholds (image px) of the K ordinal heads; HEAD3: int, index of the 3 px head
THRESH = np.array([1.5, 2.0, 3.0, 4.0, 5.0])
HEAD3 = 2
# CLIP_Z: float, standardised features are clipped to [-CLIP_Z, CLIP_Z]
CLIP_Z = 8.0
DEV = "cuda" if torch.cuda.is_available() else "cpu"


class Standardiser:
    """z = clip((x - mu) / sd). mu, sd: (F,) float64 from the fitting rows."""

    def __init__(self, X):
        self.mu = X.mean(0)
        self.sd = X.std(0) + 1e-6

    def __call__(self, X):
        return np.clip((X - self.mu) / self.sd, -CLIP_Z, CLIP_Z).astype(np.float32)


def targets(err, heads):
    """err (n,) float64 -> (n, len(heads)) float32 indicators err > THRESH[heads]."""
    return (err[:, None] > THRESH[heads][None]).astype(np.float32)


def fit_lr(Z, Y, l2):
    """Logistic regression with K independent heads. Z (n, F) float32, Y (n, K) -> nn.Linear(F, K) on CPU."""
    Zt, Yt = torch.from_numpy(Z).double(), torch.from_numpy(Y).double()
    lin = nn.Linear(Z.shape[1], Y.shape[1]).double()
    nn.init.zeros_(lin.weight)
    nn.init.zeros_(lin.bias)
    opt = torch.optim.LBFGS(lin.parameters(), lr=1.0, max_iter=500, history_size=20, line_search_fn="strong_wolfe")
    bce = nn.BCEWithLogitsLoss()

    def closure():
        opt.zero_grad()
        loss = bce(lin(Zt), Yt) + l2 * (lin.weight ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    return lin.float().eval()


def make_mlp(f, k, hidden):
    """MLP F -> hidden -> hidden -> K (SiLU). Params: (F+1)h + (h+1)h + (h+1)K."""
    return nn.Sequential(nn.Linear(f, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, k))


def fit_mlp(Z, Y, wd, hidden=32, epochs=15, seeds=(0, 1, 2), batch=1024):
    """Ensemble of small MLPs (AdamW, cosine LR). Returns list of nn.Module (CPU, eval)."""
    Zt, Yt = torch.from_numpy(Z).to(DEV), torch.from_numpy(Y).to(DEV)
    nets = []
    for s in seeds:
        torch.manual_seed(s)
        net = make_mlp(Z.shape[1], Y.shape[1], hidden).to(DEV)
        steps = epochs * ((len(Z) + batch - 1) // batch)
        opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=wd)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=steps, pct_start=0.1)
        bce = nn.BCEWithLogitsLoss()
        g = torch.Generator(device="cpu").manual_seed(s)
        for _ in range(epochs):
            perm = torch.randperm(len(Z), generator=g).to(DEV)
            for i in range(0, len(Z), batch):
                b = perm[i:i + batch]
                loss = bce(net(Zt[b]), Yt[b])
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
        nets.append(net.cpu().eval())
    return nets


def predict(model, Z, head):
    """Model (Linear or list of MLPs) -> (n,) float64 score = -logit of head (mean over the ensemble)."""
    with torch.no_grad():
        Zt = torch.from_numpy(Z)
        if isinstance(model, list):
            lg = torch.stack([m(Zt) for m in model]).mean(0)
        else:
            lg = model(Zt)
        lg = lg.double().numpy()
    if head == "mean":
        return -lg.mean(1)
    return -lg[:, head]


def fit(kind, X, err, hp):
    """Fit one model. kind 'lr' | 'mlp'; X (n, F) float32 raw features; err (n,) -> (standardiser, model)."""
    st = Standardiser(X)
    Z = st(X)
    heads = np.arange(len(THRESH)) if hp["multi"] else np.array([HEAD3])
    Y = targets(err, heads)
    model = fit_lr(Z, Y, hp["l2"]) if kind == "lr" else fit_mlp(Z, Y, hp["wd"], hidden=hp.get("hidden", 32))
    return st, model


def score(kind, st, model, X, hp):
    head = (hp.get("head", HEAD3) if hp["multi"] else 0)
    return predict(model, st(X), head)


def load_all():
    tr, tn, va = C.load_set("train"), C.load_set("trainnoisy"), C.load_set("val")
    return tr, tn, va


def train_cv(kind, hp, tr, tn, use_noisy=True, cols=None):
    """5-fold grouped CV on the train sample -> fold-out clean train scores (20000,) float64."""
    sel = slice(None) if cols is None else cols
    out = np.empty(C.N_TRAIN)
    for f in range(5):
        fit_tr = C.TRAIN_FOLD != f
        Xs, es = [tr["feats"][fit_tr][:, sel]], [tr["err"][fit_tr]]
        if use_noisy:
            m = C.TRAIN_FOLD[tn["src"]] != f
            Xs.append(tn["feats"][m][:, sel])
            es.append(tn["err"][m])
        st, model = fit(kind, np.concatenate(Xs), np.concatenate(es), hp)
        out[~fit_tr] = score(kind, st, model, tr["feats"][~fit_tr][:, sel], hp)
    return out


# SMALL3: list[int], 3 strong features for a low-capacity LR (9x9 mass, entropy, background floor / peak)
SMALL3 = [ft.FEATURE_NAMES.index(n) for n in ("lgt_m9", "entropy", "log_floor_over_peak")]
# NOPOS: list[int], all features except tip position and raw logit scale (possible train -> val shift)
NOPOS = [j for j, n in enumerate(ft.FEATURE_NAMES) if n not in ("pos_x", "pos_y", "pos_r", "lmax_raw", "lse_raw")]


def cv_metrics(s, err):
    bad = err > 3.0
    return ev.auroc(s, bad), ev.average_precision(s, bad), ev.auroc(s, err > 2.0)


def run_cv(only=None):
    """Train-CV grid -> printed table. only: None (all) or 'lr' / 'mlp' / 'small' (part of the grid)."""
    tr, tn, va = load_all()
    err = tr["err"]
    print(f"train clean bad3={int((err > 3).sum())}, noisy bad3={int((tn['err'] > 3).sum())} of {len(tn['err'])}")
    print("univariate train AUROC>3px (sign-free, 0.5 = useless), top 15:")
    uni = []
    for j, n in enumerate(ft.FEATURE_NAMES):
        a = ev.auroc(tr["feats"][:, j].astype(np.float64), err > 3)
        uni.append((max(a, 1 - a), n, a))
    for a, n, raw in sorted(uni, reverse=True)[:15]:
        print(f"   {n:<22} {raw:.4f}")
    print(f"{'config':<52} {'AUROC>3':>8} {'AP>3':>7} {'AUROC>2':>8}")
    for name, s in (("conf", np.exp(tr["feats"][:, ft.FEATURE_NAMES.index('log_m5')]).astype(np.float64)),
                    ("m9", np.exp(tr["feats"][:, ft.FEATURE_NAMES.index('log_m9')]).astype(np.float64))):
        print(f"{name:<52} " + " ".join(f"{v:8.4f}" for v in cv_metrics(s, err)))
    grid = []
    for l2 in (1e-4, 1e-3, 1e-2):
        for multi in (False, True):
            for noisy in (False, True):
                grid.append(("lr", {"l2": l2, "multi": multi}, noisy))
    for wd in (1e-3, 1e-2, 1e-1):
        for multi in (False, True):
            grid.append(("mlp", {"wd": wd, "multi": multi}, True))
    grid.append(("mlp", {"wd": 1e-2, "multi": True}, False))
    grid.append(("lr", {"l2": 1e-3, "multi": True, "head": "mean"}, True))
    grid.append(("mlp", {"wd": 1e-2, "multi": True, "head": "mean"}, True))
    # small LR: a few hand-picked strong features (fewer weights -> less overfit / train->val shift)
    for cols in (SMALL3, SMALL3 + [ft.FEATURE_NAMES.index("lgt_m5_minus_ceil")], NOPOS):
        grid.append(("lr", {"l2": 1e-3, "multi": True, "cols": cols}, True))
    for kind, hp, noisy in grid:
        if only is not None and not (kind == only or (only == "small" and "cols" in hp)):
            continue
        t0 = time.time()
        s = train_cv(kind, hp, tr, tn, use_noisy=noisy, cols=hp.get("cols"))
        tag = {k: v for k, v in hp.items() if k != "cols"} | ({"ncols": len(hp["cols"])} if "cols" in hp else {})
        print(f"{kind + ' ' + str(tag) + (' +noisy' if noisy else ''):<52} "
              + " ".join(f"{v:8.4f}" for v in cv_metrics(s, err)) + f"   {time.time() - t0:.0f}s", flush=True)


def fit_rows(tr, tn, va=None, va_mask=None, cols=None):
    """Stack the fitting rows: clean train + noisy train (+ val rows va_mask) -> X (n, F), err (n,)."""
    sel = slice(None) if cols is None else cols
    Xs, es = [tr["feats"][:, sel], tn["feats"][:, sel]], [tr["err"], tn["err"]]
    if va is not None:
        Xs.append(va["feats"][va_mask][:, sel])
        es.append(va["err"][va_mask])
    return np.concatenate(Xs), np.concatenate(es)


def run_final(kind, hp, tag, cols=None):
    """Both protocols for one config -> CACHE/raw_<tag>_{trainfit,xfit}.npz (val (22500,), train (20000,))."""
    tr, tn, va = load_all()
    sel = slice(None) if cols is None else cols
    t0 = time.time()
    st, model = fit(kind, *fit_rows(tr, tn, cols=cols), hp)
    sv = score(kind, st, model, va["feats"][:, sel], hp)
    s_tr = train_cv(kind, hp, tr, tn, cols=cols)                       # fold-out train scores
    np.savez(C.CACHE / f"raw_{tag}_trainfit.npz", val=sv, train=s_tr)
    # shipping model = this trainfit model (train + noisy only): val stays free for its isotonic calibration
    names = list(np.array(ft.FEATURE_NAMES)[sel])
    save_model(kind, hp, tag, st, model, names)
    sv, s_tr = np.empty(C.N_VAL), np.zeros(C.N_TRAIN)
    for a in (0, 1):
        st, model = fit(kind, *fit_rows(tr, tn, va, C.VAL_FOLD == a, cols=cols), hp)
        sv[C.VAL_FOLD != a] = score(kind, st, model, va["feats"][C.VAL_FOLD != a][:, sel], hp)
        s_tr += score(kind, st, model, tr["feats"][:, sel], hp) / 2    # in-sample train scores
    np.savez(C.CACHE / f"raw_{tag}_xfit.npz", val=sv, train=s_tr)
    print(f"{tag}: done {time.time() - t0:.0f}s", flush=True)


def save_model(kind, hp, tag, st, model, names):
    """Write the fitted model: LR -> model_<tag>.json (repo, tiny); MLP -> CACHE/model_<tag>.pth."""
    if kind == "lr":
        w = model.weight.detach().double().numpy()
        b = model.bias.detach().double().numpy()
        h = hp.get("head", HEAD3) if hp["multi"] else 0
        C.save_json(C.HERE / f"model_{tag}.json", {
            "what": "score = -(z . w + b), z = clip((feats - mu) / sd, -8, 8); feats = features.map_features",
            "features": names, "mu": st.mu.tolist(), "sd": st.sd.tolist(),
            "w": w[h].tolist(), "b": float(b[h]), "threshold_px": float(THRESH[HEAD3]), "hp": hp})
    else:
        torch.save({"mu": st.mu, "sd": st.sd, "nets": [m.state_dict() for m in model], "hp": hp, "features": names},
                   C.CACHE / f"model_{tag}.pth")


def run_timing():
    """CPU ms per image (1 thread, batch 1) of map_features and of the LR / MLP heads."""
    torch.set_num_threads(1)
    lg = torch.from_numpy(np.ascontiguousarray(C.load_set("val", ("logp",))["logp"][:100])).float()
    for i in range(10):
        ft.map_features(lg[i:i + 1])
    t0 = time.perf_counter()
    for i in range(100):
        ft.map_features(lg[i:i + 1])
    t_feat = 1000 * (time.perf_counter() - t0) / 100
    z = torch.zeros(1, len(ft.FEATURE_NAMES))
    lin = nn.Linear(len(ft.FEATURE_NAMES), 1)
    mlps = [make_mlp(len(ft.FEATURE_NAMES), len(THRESH), 32) for _ in range(3)]
    with torch.no_grad():
        t0 = time.perf_counter()
        for _ in range(1000):
            lin(z)
        t_lr = (time.perf_counter() - t0)
        t0 = time.perf_counter()
        for _ in range(1000):
            [m(z) for m in mlps]
        t_mlp = (time.perf_counter() - t0)
    print(f"map_features: {t_feat:.2f} ms / image (torch CPU, 1 thread);  LR head {t_lr:.3f} ms;  "
          f"3-MLP head {t_mlp:.3f} ms;  params LR {len(ft.FEATURE_NAMES) + 1}, "
          f"MLP x3 {3 * sum(p.numel() for p in mlps[0].parameters())}")


# FINAL_CONFIGS: tag -> (kind, hp, cols), the configs fixed by the train CV (see SUMMARY.md)
FINAL_CONFIGS = {
    # train CV AUROC>3 / AP>3 (float64 features); conf 0.9405 / 0.143, m9 0.9384 / 0.117
    "lr36": ("lr", {"l2": 1e-3, "multi": True}, NOPOS),         # 0.9517 / 0.160, best of the grid
    "lr3": ("lr", {"l2": 1e-3, "multi": True}, SMALL3),         # 0.9508 / 0.138, 4 weights per head
    "mlp36": ("mlp", {"wd": 1e-2, "multi": True}, NOPOS),       # MLPs lost to LR in CV (~0.943); for the record
}

if __name__ == "__main__":
    if sys.argv[1:2] == ["cv"]:
        run_cv(sys.argv[2] if len(sys.argv) > 2 else None)
    elif sys.argv[1:] == ["timing"]:
        run_timing()
    elif sys.argv[1:2] == ["final"]:
        for tag in (sys.argv[2:] or FINAL_CONFIGS):
            run_final(*FINAL_CONFIGS[tag][:2], tag, cols=FINAL_CONFIGS[tag][2])
    else:
        print(__doc__)
