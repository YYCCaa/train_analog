# Model B: Depthwise Separable

## Architecture

| Property | Value |
|----------|-------|
| Type | DW-separable encoder-decoder with skip-add |
| Input | 256x256x3 RGB |
| Encoder | regular 3->16, DW-sep 16->32->64 (stride-2, down to 32x32) |
| Bottleneck | DW-sep 64->64 (x2) |
| Decoder | DW-sep 64->32->16 (bilinear upsample + skip-add) |
| Head | 1x1 conv -> sigmoid |
| Output | 256x256x1 heatmap |
| Parameters | ~17K |

## Data

Generate **5000** training images (256x256, same dataset as Model A):

```bash
# In generate_synthetic.py, set NUM_IMAGES = 5000
python generate_synthetic.py
```

## Training Command

```bash
python train_b_dw_sep.py \
  --data-dir synthetic_data \
  --epochs 60 \
  --batch-size 32 \
  --lr 1e-3 \
  --weight-decay 1e-4 \
  --sigma 3.0
```

## Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Epochs | 60 | |
| Batch size | 32 | |
| Learning rate | 1e-3 | Lower than Model A; DW-sep can be less stable at high LR |
| Weight decay | 1e-4 | |
| Sigma | 3.0 | |
| Scheduler | CosineAnnealing | |
| Loss | MSE | |
| Optimizer | Adam | |
| Val split | 15% | |

## Benchmark Only

```bash
python train_b_dw_sep.py --benchmark-only --save-onnx model_b_dw_sep.onnx
```

## Expected Results

- Val distance: < 3px with 5000 training images
- Fewest FLOPs of the three models; wider channels (64) give more feature capacity than Model A despite fewer total params
- Benchmark runs 1-thread; multiply by ~1.4 for i5-11400 estimate if running on i9-13900K
