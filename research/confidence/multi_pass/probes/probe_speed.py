"""Probe: model throughput, inside_dial and noise_mask CPU cost (multi_pass research). Run from the repo root."""
import sys, time
sys.path.insert(0, r"C:\Users\ADMINI~1\AppData\Local\Temp\claude\C--Users-Administrator-PycharmProjects-train-dial\f292a1d6-f81e-4d2c-b353-1ee00d6e9308\scratchpad\confbundle")
import numpy as np, torch
import confbundle as cb
val = cb.load_split("val")
print({k: (v.shape, v.dtype) for k, v in val.items()})
model = cb.load_model()
cb.logits(model, val["images"][:512])
torch.cuda.synchronize(); t = time.time()
lg = cb.logits(model, val["images"][:2048])
torch.cuda.synchronize(); print("img/s", 2048 / (time.time() - t))
t = time.time()
for i in range(20): cb.inside_dial(val["images"][i])
print("inside_dial ms", (time.time() - t) / 20 * 1000)
rng = np.random.default_rng(0)
t = time.time()
for i in range(20): cb.noise_mask(rng)
print("noise_mask ms", (time.time() - t) / 20 * 1000)
