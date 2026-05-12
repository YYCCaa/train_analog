# Robust Tiny Heatmap Architectures: Design Analysis & Proposals

## 1. Diagnosis: Why Models A and C Fail Under Heavy Augmentation

### 1.1 The Bottleneck Capacity Problem

The single most important finding is the **feature capacity at the bottleneck**:

| Model | Bottleneck Channels | Spatial Size | Total Capacity | Converges? |
|-------|--------------------:|-------------:|---------------:|:----------:|
| A (MicroUNet)  | 32  | 32x32 | 32,768  | NO  |
| B (DWSepNet)   | 128 | 32x32 | 131,072 | YES |
| C (DirectNet)  | 32  | 16x16 | 8,192   | NO  |

Model B has **4x the bottleneck capacity** of A and **16x that of C**. Under heavy
augmentation (HSV jitter, shadows, pointer tip erasure, triangular artifacts,
noise, blur), the network must learn a representation that is invariant to all
these corruptions while still encoding the precise pointer tip location.
A 32-channel bottleneck simply cannot encode enough variation to do this.

### 1.2 Why Width Matters More Than Depth for Augmentation Robustness

Research on network width vs. depth shows that **wider networks produce smoother
loss landscapes**, making optimization easier under noisy/augmented training
conditions. Key findings:

- Shallower layers converge faster than deeper layers; flatter local minima in
  shallow layers produce more stable gradients.
- Wider networks need stronger regularization (weight decay) to fully benefit,
  but the base optimization landscape is inherently smoother.
- At very small scales (< 100K params), **width is the primary knob** for
  model capacity. A 32-channel bottleneck gives the model only 32 orthogonal
  feature directions to encode all spatial information -- far too few when the
  input distribution is broadened by heavy augmentation.

**Minimum effective width**: Based on Model B's success at 128 bottleneck
channels and Model A's failure at 32, the minimum effective bottleneck width
for this task under heavy augmentation is approximately **64-96 channels**.
DWSep convolutions allow reaching this width at a fraction of the parameter
cost of regular convolutions.

### 1.3 Additional Failure Factors

**Model A** (21K params):
- Single conv per encoder level provides no within-level refinement
- Additive skips between mismatched representations (single-conv features are
  not well-structured enough for effective addition)
- 32-channel bottleneck is too narrow

**Model C** (40K params):
- No skip connections at all -- the 16x16x32 bottleneck is a hard information
  bottleneck through which ALL spatial information must pass
- 4 levels of stride-2 downsampling compress to 16x16, which is only 0.4% of
  the original spatial area
- Despite having 40K params, they are spread across 4 encoder + 3 decoder
  levels rather than concentrated where they matter (bottleneck width)

### 1.4 What Model B Gets Right

Model B's success comes from three factors working together:

1. **Wide channels via DWSep**: 128 at bottleneck with only 61K total params
   (DWSep keeps parameter count low while allowing wide feature maps)
2. **Moderate downsampling**: Only 3 levels (32x32 minimum), preserving more
   spatial information
3. **Additive skip connections**: Provide gradient highways and spatial detail
   to the decoder

## 2. Research-Backed Design Principles

### 2.1 Architectural Patterns for Convergence Under Heavy Augmentation

From the pose estimation and keypoint detection literature:

- **Residual connections in bottleneck blocks** are critical. The Stacked
  Hourglass Network attributes much of its robustness to residual modules
  within the encoder-decoder, not just skip connections between them.
  Pre-activation residual blocks (BN-ReLU-Conv ordering) are more stable for
  small models.

- **Intermediate supervision** (loss at multiple decoder stages) helps
  convergence by providing shorter gradient paths. Even for a single
  keypoint, applying the loss at the bottleneck resolution (upsampled) plus
  the final resolution improves training stability.

- **Inverted residual blocks** (MobileNetV2 pattern: thin-expand-DW-project-thin)
  with linear bottlenecks provide excellent parameter efficiency while
  maintaining gradient flow through the residual connection.

### 2.2 CoordConv for Spatial Tasks

The CoordConv paper (Uber, NeurIPS 2018) demonstrates that standard convolutions
have an "intriguing failing" -- they struggle to learn coordinate-dependent
mappings. For heatmap regression, where the output is explicitly a spatial
coordinate encoded as a Gaussian:

- CoordConv adds 2 channels (normalized x, y coordinates) to the input tensor
- Cost: (C+2) * C_out * k^2 instead of C * C_out * k^2 (negligible increase)
- Benefit: 150x faster convergence on coordinate tasks in the original paper
- **Recommendation**: Use CoordConv only at the first encoder layer (where
  spatial position information enters the network) -- applying it everywhere
  adds parameters without proportional benefit at tiny model scales

### 2.3 Squeeze-and-Excitation (SE) Channel Attention

SE blocks recalibrate channel-wise features using global average pooling
followed by a lightweight two-FC gating mechanism:

- Negligible parameter cost: 2 * C * (C/r) parameters per block (with
  reduction ratio r=4, this is C^2/2 parameters)
- For 96-channel features: only 2 * 96 * 24 = 4,608 params per SE block
- Helps the network suppress noise-related features and amplify
  pointer-relevant features -- directly addresses the augmentation problem
- Research shows SE is particularly effective in small models where every
  channel must be maximally informative

### 2.4 Decoder Design for Precise Heatmap Localization

Key findings from the literature:

- **Bilinear upsample + conv is superior to transposed convolution** at this
  scale: avoids checkerboard artifacts, fewer parameters, cleaner ONNX export
- **Concatenation skips preserve more information than additive skips** but
  require a 1x1 projection layer to reduce channels after concatenation
- **Pixel shuffle (sub-pixel convolution)** offers an alternative that learns
  upsampling parameters but adds complexity; bilinear + conv is simpler and
  sufficient for our heatmap output resolution
- **Half-resolution output (128x128) is strongly recommended**: at sigma=1.5
  (scaled from 3.0), the Gaussian peak is still well-resolved and the
  decoder requires one fewer upsample stage

## 3. Proposed Architectures

### 3.1 Summary Comparison

| Model | Params | MACs | P95 Latency* | Key Innovation |
|-------|-------:|-----:|-------------:|----------------|
| A (existing)   | 21K  |  62M | 2.2ms | Baseline U-Net |
| B (existing)   | 61K  | 166M | 4.4ms | Wide DWSep (converges) |
| C (existing)   | 40K  |  67M | 1.7ms | No skips (fails) |
| **D (proposed)** | **52K** | -- | **5.4ms** | CoordConv + concat skips + residual bottleneck |
| **E (proposed)** | **71K** | -- | **5.6ms** | MobileNetV2 inverted residuals + linear bottleneck |
| **F (proposed)** | **65K** | -- | **8.1ms** | SE attention + DWSep + residual bottleneck |

*P95 latency measured with ONNX Runtime, 1 thread, on the current machine.
Multiply by ~1.2-1.5 for i5-11400 estimate. All are well under 30ms.

---

### 3.2 Model D: Wide Residual DWSep with CoordConv + Concatenation Skips

**52K parameters | ~5ms P95 inference**

**Rationale**: Combines CoordConv (for spatial awareness in heatmap regression),
concatenation-based skip connections (preserving more encoder information than
additive skips), and residual DWSep bottleneck blocks (for gradient flow and
augmentation robustness). The wider channels (96 at bottleneck) match Model B's
proven approach while CoordConv compensates for any capacity gap.

```
Layer-by-layer specification:

ENCODER:
  enc1: CoordConv(3+2 -> 32, k=3, s=2, p=1) + BN + ReLU     -> 128x128x32
  enc2: DWSep(32 -> 64, k=3, s=2, p=1)                        -> 64x64x64
         DW: Conv2d(32, 32, 3, s=2, groups=32) + BN + ReLU
         PW: Conv2d(32, 64, 1) + BN + ReLU
  enc3: DWSep(64 -> 96, k=3, s=2, p=1)                        -> 32x32x96
         DW: Conv2d(64, 64, 3, s=2, groups=64) + BN + ReLU
         PW: Conv2d(64, 96, 1) + BN + ReLU

BOTTLENECK (2x pre-activation residual DWSep):
  bn1:  BN(96) + ReLU + DW(96, k=3, groups=96) + BN + ReLU + PW(96->96) + residual add
  bn2:  BN(96) + ReLU + DW(96, k=3, groups=96) + BN + ReLU + PW(96->96) + residual add
                                                               -> 32x32x96

DECODER (concatenation skips):
  up3:  Bilinear upsample 32->64                               -> 64x64x96
        Concat with e2                                         -> 64x64x160
        1x1 Conv(160->64) + BN + ReLU (channel reduce)
        DWSep(64->64)                                          -> 64x64x64

  up2:  Bilinear upsample 64->128                              -> 128x128x64
        Concat with e1                                         -> 128x128x96
        1x1 Conv(96->32) + BN + ReLU (channel reduce)
        DWSep(32->32)                                          -> 128x128x32

HEAD:
  1x1 Conv(32->1) + Sigmoid                                   -> 128x128x1
  (If full-res: bilinear 2x upsample to 256x256)
```

**Why it should handle heavy augmentation**:
- CoordConv at input gives the network direct access to pixel coordinates,
  reducing the learning burden for a task whose output is inherently
  coordinate-dependent
- 96 bottleneck channels provide 3x the capacity of Model A
- Pre-activation residual connections in the bottleneck stabilize gradients
  and prevent vanishing gradients during early training under noisy data
- Concatenation skips preserve full encoder features without the alignment
  issues of additive skips (important when augmentation changes feature
  statistics)

**Parameter breakdown**:
- enc1 (CoordConv): 1,440 + 64 (BN) = 1,504
- enc2: 288 + 2,048 = ~2,400
- enc3: 576 + 6,144 = ~6,800
- Bottleneck (2 blocks): ~3,700
- Decoder: ~30,000 (dominated by 1x1 concat-reduction layers)
- Head: 33
- Total: ~52K

---

### 3.3 Model E: Inverted Residual Hourglass

**71K parameters | ~5.6ms P95 inference**

**Rationale**: Uses MobileNetV2-style inverted residual blocks throughout.
These blocks expand channels internally (thin->wide->DW->thin), providing
rich intermediate representations while keeping skip-connection channels thin.
The "linear bottleneck" (no activation after the final 1x1 projection)
preserves information better than ReLU-capped outputs, which is critical for
the precise heatmap regression task.

```
Layer-by-layer specification:

STEM:
  Conv2d(3 -> 24, k=3, s=2, p=1) + BN + ReLU6               -> 128x128x24

ENCODER:
  enc2: InvRes(24 -> 48, s=2, expand=2)                       -> 64x64x48
         Expand: Conv2d(24, 48, 1) + BN + ReLU6
         DW:     Conv2d(48, 48, 3, s=2, groups=48) + BN + ReLU6
         Project: Conv2d(48, 48, 1) + BN (LINEAR, no activation)

  enc3: InvRes(48 -> 64, s=2, expand=2)                       -> 32x32x64
         Expand: Conv2d(48, 96, 1) + BN + ReLU6
         DW:     Conv2d(96, 96, 3, s=2, groups=96) + BN + ReLU6
         Project: Conv2d(96, 64, 1) + BN (LINEAR)

BOTTLENECK (2x InvRes with residual):
  bn1:  InvRes(64 -> 64, s=1, expand=2) + residual add        -> 32x32x64
         Expand: Conv2d(64, 128, 1) + BN + ReLU6
         DW:     Conv2d(128, 128, 3, groups=128) + BN + ReLU6
         Project: Conv2d(128, 64, 1) + BN (LINEAR)
  bn2:  same as bn1                                            -> 32x32x64

DECODER (additive skips with 1x1 projection):
  up3:  Bilinear upsample 32->64                               -> 64x64x64
        1x1 Conv(64->48) + BN (project to match e2)
        Add with e2                                            -> 64x64x48
        InvRes(48->48, expand=2) + residual                    -> 64x64x48

  up2:  Bilinear upsample 64->128                              -> 128x128x48
        1x1 Conv(48->24) + BN (project to match e1)
        Add with e1                                            -> 128x128x24
        InvRes(24->24, expand=2) + residual                    -> 128x128x24

HEAD:
  1x1 Conv(24->1) + Sigmoid                                   -> 128x128x1
  (If full-res: bilinear 2x upsample to 256x256)
```

**Why it should handle heavy augmentation**:
- Inverted residuals expand to 128 channels internally at the bottleneck,
  giving the model access to a very wide representation for computing
  features, then compress back to 64 for the residual connection
- Linear bottleneck (no ReLU after projection) prevents information loss
  that would compound under noisy training -- ReLU6 clips extreme
  activations (important under heavy augmentation)
- Every block has a residual connection (when in/out match), providing
  robust gradient flow throughout the entire network
- The expand-then-compress pattern means the model uses parameters efficiently:
  the expensive DW conv operates on expanded channels but adds no cross-channel
  cost, and the cheap 1x1 projections handle channel mixing

**Parameter breakdown**:
- Stem: 696
- enc2: 4,176
- enc3: 12,128
- Bottleneck (2 blocks): 36,352
- Decoder projections: 4,368
- Decoder InvRes blocks: 13,536
- Head: 25
- Total: ~71K

---

### 3.4 Model F: SE-Enhanced DWSep (Model B+)

**65K parameters | ~8ms P95 inference**

**Rationale**: This is the **conservative, lowest-risk** proposal. It takes
Model B's proven architecture (which already converges!) and adds two specific
improvements: (1) Squeeze-and-Excitation channel attention after each DWSep
block, and (2) residual connections in the bottleneck. The channel progression
is slightly adjusted (32->64->96 instead of 32->64->128) since SE attention
makes each channel more effective. This is the architecture most likely to
converge on the first attempt because it is closest to what already works.

```
Layer-by-layer specification:

ENCODER:
  enc1: Conv2d(3 -> 32, k=3, s=2, p=1) + BN + ReLU           -> 128x128x32

  enc2: DWSepSE(32 -> 64, s=2)                                -> 64x64x64
         DW: Conv2d(32, 32, 3, s=2, groups=32) + BN + ReLU
         PW: Conv2d(32, 64, 1) + BN + ReLU
         SE: GAP -> FC(64->16) -> ReLU -> FC(16->64) -> Sigmoid -> scale

  enc3: DWSepSE(64 -> 96, s=2)                                -> 32x32x96
         DW: Conv2d(64, 64, 3, s=2, groups=64) + BN + ReLU
         PW: Conv2d(64, 96, 1) + BN + ReLU
         SE: GAP -> FC(96->24) -> ReLU -> FC(24->96) -> Sigmoid -> scale

BOTTLENECK (2x residual DWSep + SE):
  bn1:  DWSep(96->96) + SE(96) + residual add                 -> 32x32x96
         DW: Conv2d(96, 96, 3, groups=96) + BN + ReLU
         PW: Conv2d(96, 96, 1) + BN + ReLU
         SE: GAP -> FC(96->24) -> ReLU -> FC(24->96) -> Sigmoid -> scale
         Output = input + SE(DWSep(input))
  bn2:  same as bn1                                            -> 32x32x96

DECODER (additive skips):
  up3:  Bilinear upsample 32->64                               -> 64x64x96
        1x1 Conv(96->64) + BN (project to match e2)
        Add with e2                                            -> 64x64x64
        DWSepSE(64->64)                                       -> 64x64x64

  up2:  Bilinear upsample 64->128                              -> 128x128x64
        1x1 Conv(64->32) + BN (project to match e1)
        Add with e1                                            -> 128x128x32
        DWSepSE(32->32)                                       -> 128x128x32

HEAD:
  1x1 Conv(32->1) + Sigmoid                                   -> 128x128x1
  (If full-res: bilinear 2x upsample to 256x256)
```

**Why it should handle heavy augmentation**:
- Directly derived from Model B which already converges -- lowest experiment
  risk of all three proposals
- SE attention recalibrates channels globally: under heavy augmentation, some
  channels will fire on noise/artifacts; SE learns to suppress these and
  amplify channels that detect the actual pointer structure
- Residual bottleneck blocks (missing from Model B) add gradient highways
  through the deepest part of the network, where gradients are weakest
- 96 bottleneck channels (vs B's 128) compensated by SE making each channel
  more informative through learned recalibration

**Parameter breakdown**:
- enc1: 928
- enc2 (DWSepSE): 4,576
- enc3 (DWSepSE): 11,648
- Bottleneck (2 blocks): 30,144
- Decoder projections: 8,384
- Decoder DWSepSE blocks: 8,928
- Head: 33
- Total: ~65K

## 4. Recommendations

### 4.1 Architecture Priority Order

1. **Model F (SE-Enhanced DWSep)** -- Train first. Lowest risk because it
   builds directly on Model B's proven convergence. If this converges with
   better accuracy than B, it validates the SE + residual bottleneck approach.

2. **Model D (CoordConv + Concat Skips)** -- Train second. The CoordConv
   input and concatenation skips represent a different architectural philosophy
   (explicit spatial information + richer skip connections). This may achieve
   better localization accuracy if it converges.

3. **Model E (Inverted Residual Hourglass)** -- Train third. Most
   architecturally novel. The MobileNetV2 inverted residual pattern is proven
   at large scale but less tested at this tiny scale. If it works, it
   represents the most parameter-efficient path to wider internal
   representations.

### 4.2 Training Recommendations

Based on Model B's successful configuration and the research:

| Parameter | Recommended | Notes |
|-----------|-------------|-------|
| Learning rate | 1e-4 to 5e-4 | Lower than Models A/C used; wider models need gentler LR |
| Weight decay | 1e-4 to 3e-4 | Increase slightly for wider networks |
| Epochs | 150 | Allow full convergence |
| Batch size | 64 | Larger batches smooth gradient noise from augmentation |
| Sigma | 3.0 (1.5 at half-res) | Keep consistent with existing setup |
| Loss | MSE + peak sharpness (weight=3e-5) | Use v2 loss from train_utils_v2.py |
| Scheduler | CosineAnnealing | Proven effective |
| Warmup | 5 epochs linear warmup to peak LR | Helps wider models avoid early instability |
| Half-res output | YES (128x128) | Fewer decoder params, faster training |

### 4.3 Techniques Worth Investigating (Future Work)

- **Intermediate supervision**: Apply auxiliary MSE loss at the bottleneck
  resolution (32x32 upsampled bilinear to 128x128) with weight 0.1-0.3.
  This provides gradient signal directly to the encoder, accelerating
  convergence under heavy augmentation.

- **Gradient clipping**: Clip gradients to max_norm=1.0 during training.
  Under heavy augmentation, occasional extreme samples can produce gradient
  spikes that destabilize small models.

- **Label smoothing for heatmaps**: Instead of pure Gaussian with peak=1.0,
  clip to max=0.95 during training. This gives the sigmoid output headroom
  and prevents extreme weight updates near saturation.

- **Progressive augmentation**: Start training with light augmentation
  (first 30 epochs) and progressively increase to full augmentation. This
  gives the model a stable initial representation before challenging it.

## 5. Sources

Research references consulted:

- [Lightweight Human Pose Estimation Using Heatmap-Weighting Loss](https://arxiv.org/pdf/2205.10611) - Lightweight deconvolution head and heatmap loss design
- [An Intriguing Failing of Convolutional Neural Networks and the CoordConv Solution](https://arxiv.org/abs/1807.03247) - CoordConv for coordinate-dependent tasks
- [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381) - Inverted residual block design
- [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507) - SE channel attention mechanism
- [Stacked Hourglass Networks for Human Pose Estimation](https://ar5iv.labs.arxiv.org/html/1603.06937) - Intermediate supervision and residual hourglass design
- [Simple Baselines for Human Pose Estimation and Tracking](https://openaccess.thecvf.com/content_ECCV_2018/papers/Bin_Xiao_Simple_Baselines_for_ECCV_2018_paper.pdf) - Simple deconvolution decoder design
- [Do Wider Neural Networks Really Help Adversarial Robustness?](https://arxiv.org/pdf/2010.01279) - Width vs. robustness trade-offs
- [Checkerboard Artifact Free Sub-pixel Convolution](https://arxiv.org/pdf/1707.02937) - Upsampling artifact analysis
- [LiPE: Lightweight Pose Estimator for Mobile Applications](https://www.sciencedirect.com/science/article/pii/S2667241324000193) - DWSep deconvolution for lightweight pose
- [Deconvolution and Checkerboard Artifacts](https://distill.pub/2016/deconv-checkerboard/) - Why bilinear+conv beats transposed conv
- [Sub-pixel Face Landmarks Using Heatmaps](https://arxiv.org/pdf/2103.03059) - Heatmap regression for precise localization
- [SD-HRNet: Lightweight Pose via Spatial Grouping and Attention](https://link.springer.com/article/10.1007/s11554-025-01834-4) - Spatial attention in lightweight pose models
