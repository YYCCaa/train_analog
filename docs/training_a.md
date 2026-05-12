# Model A: Micro U-Net

## Architecture

| Property | Value |
|----------|-------|
| Type | U-Net with additive skip connections |
| Input | 256x256x3 RGB |
| Encoder | 3 levels: 3->8->16->32 (stride-2 convs, down to 32x32) |
| Bottleneck | 32->32 (single 3x3 conv) |
| Decoder | 32->16->8 (bilinear upsample + conv + skip-add) |
| Head | 1x1 conv -> sigmoid |
| Output | 256x256x1 heatmap |
| Parameters | ~21K |

## Data

Generate **5000** training images (256x256) before running:

```bash
# In generate_synthetic.py, set NUM_IMAGES = 5000
python generate_synthetic.py
```

## Training Command

```bash
python train_a_micro_unet.py \
  --data-dir synthetic_data \
  --epochs 60 \
  --batch-size 32 \
  --lr 2e-3 \
  --weight-decay 1e-4 \
  --sigma 3.0
```

## Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Epochs | 60 | Small model converges quickly |
| Batch size | 32 | |
| Learning rate | 2e-3 | Higher LR suits tiny models |
| Weight decay | 1e-4 | Light regularisation |
| Sigma | 3.0 | Tight Gaussian heatmap |
| Scheduler | CosineAnnealing | Decays LR to ~0 over training |
| Loss | MSE | Standard for heatmap regression |
| Optimizer | Adam | |
| Val split | 15% | |

## Benchmark Only

```bash
python train_a_micro_unet.py --benchmark-only --save-onnx model_a_micro_unet.onnx
```

## Expected Results

- Val distance: < 3px with 5000 training images
- Benchmark runs 1-thread; if your CPU is an i9-13900K, multiply latency by ~1.4 to estimate i5-11400 performance
