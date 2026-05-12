# Model C: Direct Encoder-Decoder (No Skips)

## Architecture

| Property | Value |
|----------|-------|
| Type | Encoder-decoder, no skip connections |
| Input | 256x256x3 RGB |
| Encoder | 4 levels: 3->8->16->32->32 (stride-2, down to 16x16) |
| Bottleneck | 32->32 |
| Decoder | 32->32->16->8 (bilinear upsample + conv) |
| Head | 1x1 conv -> sigmoid |
| Output | 256x256x1 heatmap |
| Parameters | ~40K |

## Data

Generate **5000** training images (256x256, same dataset as Models A/B):

```bash
# In generate_synthetic.py, set NUM_IMAGES = 5000
python generate_synthetic.py
```

## Training Command

```bash
python train_c_direct.py \
  --data-dir synthetic_data \
  --epochs 80 \
  --batch-size 32 \
  --lr 2e-3 \
  --weight-decay 1e-4 \
  --sigma 3.0
```

## Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Epochs | 80 | More epochs needed; no skips means slower convergence |
| Batch size | 32 | |
| Learning rate | 2e-3 | |
| Weight decay | 1e-4 | |
| Sigma | 3.0 | |
| Scheduler | CosineAnnealing | |
| Loss | MSE | |
| Optimizer | Adam | |
| Val split | 15% | |

## Benchmark Only

```bash
python train_c_direct.py --benchmark-only --save-onnx model_c_direct.onnx
```

## Expected Results

- Val distance: < 5px (may have worse spatial precision than A/B without skip connections)
- Deeper encoding (16x16) captures more global context
- Simplest architecture; good baseline for comparing skip connection benefit
- Benchmark runs 1-thread; multiply by ~1.4 for i5-11400 estimate if running on i9-13900K
