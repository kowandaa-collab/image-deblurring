# BlurDM v4: Blur Diffusion Model for Image Deblurring

> Diffusion models show promise for dynamic scene deblurring; however, existing studies often fail to leverage the intrinsic nature of the blurring process within diffusion models, limiting their full potential. To address it, we present a Blur Diffusion Model (BlurDM), which seamlessly integrates the blur formation process into diffusion for image deblurring. Observing that motion blur stems from continuous exposure, BlurDM implicitly models the blur formation process through a dual-diffusion forward scheme, diffusing both noise and blur onto a sharp image. During the reverse generation process, we derive a dual denoising and deblurring formulation, enabling BlurDM to recover the sharp image by simultaneously denoising and deblurring, given pure Gaussian noise conditioned on the blurred image as input. Additionally, to efficiently integrate BlurDM into deblurring networks, we perform BlurDM in the latent space, forming a flexible prior generation network for deblurring.

![Pipeline](assets/BlurDM_teaser.png)

## What's New in v4

This version introduces **HybridBlurDM**, a custom hybrid backbone designed to push beyond the state-of-the-art on motion deblurring. It also adds full support for **NAFNet** as an alternative backbone, both integrated into the three-stage BlurDM training pipeline.

---

## HybridBlurDM

HybridBlurDM is a new backbone that combines the most effective ideas from recent deblurring literature into a single, parameter-efficient architecture.

**Key innovations:**

1. **DualDomainMixer** — per-block FFT branch (global, O(HW log HW)) and depthwise spatial branch (local, NAFNet-style SCA) run *in parallel* and are fused with a learned channel-wise gate. Unlike FFTformer (sequential) or NAFNet (spatial only), both domains are active simultaneously.

2. **MotionStripConv** — asymmetric depthwise convolutions (1×K horizontal + K×1 vertical) capture directional motion-blur streaks cheaply. Inspired by Stripformer but replaced with efficient DW strip convs to avoid O(HW²) strip attention cost. K=21 covers longer streak lengths at training resolution.

3. **GatedFFN** — NAFNet-style element-wise gate (a × b from split channels) instead of GELU activation, for better gradient flow.

4. **Multi-scale FiLM injection** — diffusion prior injected at *every* decoder level (not just bottleneck), giving four distinct opportunities to steer reconstruction at 1/4, 1/2, and full scale, yielding larger Stage 3 gains.

5. **Deformable bottleneck** — `DeformConv2d` at bottleneck encoder blocks with zero-initialized offsets (starts as standard conv, learns to deform). Falls back gracefully if torchvision is missing `DeformConv2d`.

**Architecture:** U-Net (4 stages, PixelShuffle down/up) with multi-scale outputs [pred_1/4, pred_1/2, pred_full].

| Variant | Backbone | LE_arch | Total |
|---------|----------|---------|-------|
| HybridBlurDM-light | ~22 M | 5.27 M | ~27 M |

**Training features:**
- Weighted multi-scale pixel loss: 0.125·L(1/4) + 0.25·L(1/2) + 0.5·L(full)
- Auxiliary full-scale losses: FFT + VGG perceptual + Haar wavelet + SSIM
- EMA shadow weights used for validation and checkpointing
- Progressive crop: starts at 128 px, switches to 256 px at epoch 100
- MixUp augmentation enabled by default

---

## Supported Backbones

| Backbone | Params | Notes |
|----------|--------|-------|
| **HybridBlurDM-light** | ~27 M | New custom architecture (this repo) |
| NAFNet | ~17 M | Nonlinear activation free network |
| MIMO_UNet | ~16 M | Original BlurDM baseline |

---

## Installation

```bash
conda create -n blurdm python=3.9
conda activate blurdm
conda install pytorch==2.0 torchvision==0.15 torchaudio==2.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install opencv-python tqdm tensorboardX pyiqa thop numpy pandas torchmetrics
```

## Dataset Preparation

Download the GoPro Large dataset from [seungjunnah.github.io](https://seungjunnah.github.io/Datasets/gopro.html).

Expected folder structure:
```
GOPRO_Large/
├─ train/
│  ├─ GOPR0372_07_00/
│  │  ├─ blur/
│  │  │  └─ *.png
│  │  └─ sharp/
│  │     └─ *.png
│  └─ ...
└─ test/
   └─ ... (same structure)
```

Other supported datasets: [HIDE](https://github.com/joanshen0508/HA_deblur), [RealBlur](https://cg.postech.ac.kr/research/realblur/)

---

## Training

Each backbone uses a **3-stage pipeline**:

| Stage | Script | What trains |
|-------|--------|-------------|
| 1 | `train_stage1.py` | Backbone + LatentEncoder jointly (oracle sharp condition) |
| 2 | `train_stage2.py` | Diffusion prior (BlurDM); loads LE weights from Stage 1 |
| 3 | `train_stage3.py` | Joint fine-tuning; loads backbone (Stage 1) + BlurDM (Stage 2) |

### HybridBlurDM — Stage 1

```powershell
python src/HybridBlurDM/train_stage1.py `
  --data_path "D:/GOPRO_Large" `
  --dir_path "experiments/HybridBlurDM/GoPro/stage1" `
  --model_name HybridBlurDM-light --model HybridBlurDM-light `
  --num_workers 4 --batch_size 8 --crop_size 128 `
  --end_epoch 300 --validation_epoch 50 --check_point_epoch 100 `
  --amp --compile --criterion l1
```

Training tips:
- Use `--init_lr 5e-5 --warmup_epochs 10 --grad_clip 0.5` and drop `--amp`/`--compile` if you see NaN loss early on
- Progressive crop switches automatically from 128→256 px at epoch 100 (`--prog_switch_epoch`)
- EMA is always enabled in HybridTrainer; checkpoint stores EMA weights

### NAFNet — Stage 1

```powershell
python src/NAFNet/train_stage1.py `
  --data_path "D:/GOPRO_Large" `
  --dir_path "experiments/NAFNet/GoPro/stage1" `
  --model_name NAFNetBlurDM-light --model NAFNetBlurDM-light `
  --num_workers 4 --batch_size 16 --crop_size 256 `
  --end_epoch 300 --validation_epoch 50 --check_point_epoch 100 `
  --amp --compile --criterion l1
```

### MIMO_UNet — Stage 1

```powershell
python src/MIMO_UNet/train_stage1.py `
  --data_path "D:/GOPRO_Large" `
  --dir_path "experiments/MIMO_UNet/GoPro/stage1" `
  --num_workers 4 --batch_size 16 --crop_size 256 `
  --end_epoch 300 --validation_epoch 50 --check_point_epoch 100
```

---

## Evaluation

```powershell
python predict.py `
  --data_path "D:/GOPRO_Large" `
  --model_path "experiments/HybridBlurDM/GoPro/stage3/best_deblur_HybridBlurDM-light.pth" `
  --output_dir "results/HybridBlurDM/GoPro"
```

Or with DDP:
```powershell
torchrun --nproc_per_node=2 predict.py ...
```

---

## Results (GoPro test set)

| Model | Stage | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|-------|-------|--------|--------|---------|
| MIMO_UNet | Stage 1 (baseline) | 25.69 | 0.857 | 0.282 |
| NAFNet | Stage 3 | 31.71 | — | 0.119 |
| **HybridBlurDM-light** | **Stage 3** | **TBD** | **TBD** | **TBD** |

> NAFNet Stage 3 establishes the in-project baseline at 31.71 dB. HybridBlurDM-light is currently in Stage 1 training.

![Results](assets/BlurDM_results.png)

---

## Acknowledgement

Training utilities built on [BasicSR](https://github.com/XPixelGroup/BasicSR). Architecture inspired by NAFNet, Restormer, FFTformer, and Stripformer.

## Citation

```bibtex
@inproceedings{heblurdm,
  title={BlurDM: A Blur Diffusion Model for Image Deblurring},
  author={He, Jin-Ting and Tsai, Fu-Jen and Peng, Yan-Tsung and Chen, Min-Hung and Lin, Chia-Wen and Lin, Yen-Yu},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems}
}
```
