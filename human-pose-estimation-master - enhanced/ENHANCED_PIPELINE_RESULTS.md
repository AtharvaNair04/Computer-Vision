# Enhanced FMP Pipeline: Method and Verified Results

Project: *Improving Classical Human Pose Estimation Using Image Preprocessing and Adaptive Feature Extraction* (Group 1).
Base model: Flexible Mixtures-of-Parts, Yang & Ramanan, CVPR 2011. The FMP model, its HOG features and its dynamic-programming inference are **unchanged**; every enhancement wraps around it, as the abstract proposes.

All numbers below are reproducible with the commands at the end. FMP inference is deterministic, so re-running gives identical results.

---

## 1. Evaluation protocol

| Item | Setting |
|---|---|
| Datasets | PARSE (205 test images, 14 joints) and BUFFY (276 test frames, 10 upper-body joints) |
| Conditions | clean + 4 synthetic degradations from `degrade_image.m`: dark (gamma 1.5), low contrast (L* compressed ×0.45), Gaussian noise (var 0.01), Gaussian blur (σ = 1.5) |
| Metrics | Official Yang & Ramanan PCK and APK at threshold 0.1 (PARSE) / 0.2 (BUFFY), left/right averaged |
| **Tuning split** | PARSE test images **1–100**: every operator, threshold and parameter was chosen here (`code_full/tuning/`) |
| **Held-out split** | PARSE test images **101–205**: never looked at during tuning. **These are the headline numbers.** |
| Cross-dataset | BUFFY: nothing was tuned on BUFFY |
| Significance | Paired bootstrap (5 000 resamples) 95% CI of the per-image PCK change |

---

## 2. Final pipeline (maps to the abstract's architecture diagram)

| Stage | Module | What it does |
|---|---|---|
| 1. Image-quality analysis | `image_quality.m` | Brightness, robust contrast (1–99th percentile), **Immerkær (1996) noise σ**, contrast-normalised Laplacian sharpness |
| 2. Adaptive preprocessing | `adaptive_preprocess.m` | At most one restoration per image, priority noise > blur > dark:<br>• noise σ > 6 → **non-local-means denoising** (`imnlmfilt`)<br>• sharpness < 0.014 → **Richardson-Lucy deconvolution** (Gaussian PSF σ 1.5, 8 iterations)<br>• brightness < 0.40 → **gamma 0.67**<br>• otherwise the image is left unchanged |
| 3. HOG feature pyramid | `detection/featpyramid.m` | Original FMP, unchanged |
| 4. FMP inference + TTA | `detect_tta.m` | Original FMP run on the image **and its mirror**; mirrored candidates are mapped back (x reflected, left/right parts swapped) and pooled |
| 4b. Pose voting | `pose_vote.m` | Each output joint is the score-weighted mean over near-duplicate candidates of the same person (IoU > 0.7, score within 0.25 of the best, weight exp(Δs/0.1)) |
| 5. Pose refinement | `refine_pose.m`, `apply_pose_refinement.m` | Anthropometric bone-length bounds + left/right symmetry prior (PARSE only) |
| 6–7. Visualisation / evaluation | `demo_complete.m`, `run_final_evaluation.m` | Demo and the evaluation below |

`enhanced_detect.m` chains stages 1–4.

### What was tested and rejected (tuning split, 100 images, mean PCK)

| Condition | Baseline | Rejected options | Kept |
|---|---|---|---|
| Noise | 67.4 | bilateral 69.1, Wiener 68.8, median 68.3, Gaussian σ1 69.0, previous repo version 67.4 | **NLM 70.7** |
| Blur | 66.1 | unsharp mask 65.9, 1.25× upsampling 66.3, previous repo version 65.9 | **R-L deconvolution 68.3** (it hurts clean images: 69.0 vs 71.4, so it is gated) |
| Dark | 70.9 | contrast stretch 71.6, previous repo version 70.7 | **gamma 0.67: 74.0** |
| Low contrast | 71.8 | gamma, stretch, previous repo version: all ≤ baseline | nothing (FMP's HOG block normalisation already handles global contrast) |
| Adaptive HOG | — | identical to baseline ±0.1 in every condition | kept only as an ablation row |
| All | — | pooling the unprocessed image as well (2× cost) gave only +0.1 | not used |

---

## 3. Results on the held-out PARSE images (101–205, N = 105)

### PCK (%)

| Condition | Baseline FMP | Adaptive HOG | + Preprocess | + Flip TTA / Vote | **+ Refinement (full)** | Full − Baseline [95% CI] |
|---|---|---|---|---|---|---|
| Clean | 75.37 | 75.44 | 74.83 | 74.42 | **75.10** | −0.27 [−2.93, +1.97] |
| Dark | 75.51 | 75.51 | 74.76 | 76.94 | **77.28** | +1.77 [−0.65, +4.15] |
| Low contrast | 74.76 | 74.69 | 74.49 | 74.69 | **75.10** | +0.34 [−1.90, +2.31] |
| Noise | 68.78 | 68.64 | 71.02 | 72.65 | **72.79** | **+4.01 [+0.75, +7.21]** |
| Blur | 68.91 | 68.91 | 71.16 | 73.13 | **73.40** | **+4.49 [+1.77, +7.69]** |

### APK (%)

| Condition | Baseline FMP | Adaptive HOG | + Preprocess | + Flip TTA / Vote | **Full** | Δ |
|---|---|---|---|---|---|---|
| Clean | 66.19 | 66.28 | 66.28 | 65.39 | **66.28** | +0.09 |
| Dark | 67.13 | 67.14 | 66.38 | 67.54 | **68.24** | +1.11 |
| Low contrast | 66.28 | 66.65 | 65.98 | 66.89 | **67.37** | +1.09 |
| Noise | 57.26 | 57.14 | 60.41 | 62.98 | **63.10** | **+5.84** |
| Blur | 57.43 | 57.44 | 61.01 | 64.79 | **65.30** | **+7.87** |

### Full PARSE test set (1–205, N = 205), for reference only (half of it is the tuning split)

| Condition | Baseline PCK | Full PCK | Δ PCK [95% CI] | Baseline APK | Full APK |
|---|---|---|---|---|---|
| Clean | 73.45 | 75.64 | +2.20 [+0.40, +4.01] | 64.48 | 67.76 |
| Dark | 73.24 | 75.64 | +2.40 [+0.63, +4.32] | 64.76 | 67.19 |
| Low contrast | 73.31 | 74.08 | +0.77 [−0.64, +2.09] | 64.97 | 66.78 |
| Noise | 68.08 | 71.95 | +3.87 [+1.64, +6.11] | 57.52 | 62.87 |
| Blur | 67.56 | 72.54 | +4.98 [+2.84, +7.28] | 57.21 | 64.66 |

## 4. BUFFY cross-dataset results (all 276 test frames, no BUFFY tuning)

Pipeline: preprocessing + flip TTA + pose voting. Refinement is not used here because it is defined for PARSE's 14 joints.

| Metric | Head | Shoulder | Elbow | Wrist | Hip | **Average** |
|---|---|---|---|---|---|---|
| PCK baseline | 96.0 | 97.1 | 87.5 | 67.2 | 85.3 | 86.63 |
| PCK enhanced | 96.2 | 97.1 | 90.0 | 71.4 | 90.2 | **88.99 (+2.36)** |
| APK baseline | 88.3 | 88.8 | 78.8 | 56.7 | 71.5 | 76.82 |
| APK enhanced | 88.5 | 89.1 | 82.0 | 59.8 | 77.5 | **79.40 (+2.59)** |

## 5. Cost

Single-process timing (`demo_complete`): baseline FMP ≈ 1.3 s per image, full pipeline ≈ 2.4 s. The flip pass doubles FMP inference. Quality analysis plus restoration adds about 0.13 s. Refinement and voting are negligible.

---

## 6. Honest reading of the results

* **Supported claims (held-out, CI excludes 0):** the pipeline makes FMP substantially more robust to **sensor noise (+4.0 PCK / +5.8 APK)** and **blur (+4.5 PCK / +7.9 APK)**.
* **Positive but not significant on 105 images:** dark (+1.8 PCK, +1.1 APK) and low contrast (+0.3 PCK, +1.1 APK).
* **No clean-image regression:** −0.3 PCK / +0.1 APK, well within noise.
* **Cross-dataset:** +2.4 PCK / +2.6 APK on BUFFY, with no tuning there.
* Flip TTA + voting improved clean images on the tuning split (+3 PCK) but not on the held-out split. Its benefit is therefore mainly under degradation, not a general accuracy gain.
* **Adaptive HOG** (`adaptive_features`) does not change accuracy. Without retraining the FMP filters on the new feature distribution, it cannot. Retraining (`train_adaptive_fmp.m`) needs the full INRIA negative set (frames 615–1832), which is not in this repository.

### Limitations

* The degradations are synthetic and at one severity each. The deblurring PSF (σ 1.5) matches the synthetic blur, so real motion blur would need PSF estimation.
* The thresholds come from 100 PARSE images. BUFFY shows they transfer, but a real-image test is still missing.
* The 2× inference cost comes from the flip pass. Preprocessing alone (no TTA) is almost free and still gives +2.2 PCK on noise and blur.

### Corrections to earlier project reports

`COMPLETE_PROJECT_WALKTHROUGH_AND_VERIFICATION.md` and `PROJECT_CHECKPOINT_REPORT.md` report numbers that cannot be reproduced. The "+50 pp on noise" came from a `complete_evaluation_results.mat` written by a run on a **single image** (every per-joint value is 0, 50 or 100). Use the tables in this document instead.

---

## 7. How to reproduce

```matlab
cd 'human-pose-estimation-master - enhanced/code_full'

% PARSE, held-out split (about 45 min single-process; cached in cache/final/)
results = run_final_evaluation(101:205);

% Optional: shard detection over several MATLAB processes first, e.g. 5 of them:
%   run_final_evaluation(1:205, 'shard', [k 5])   % k = 1..5, one per process
% then call run_final_evaluation(...) once more for the metrics.

% BUFFY, all 276 test frames
buffy = run_buffy_experiments();

% Interactive demo
demo_complete(4, 'noise');
demo_complete(1, 'blur');
```

Figures are written to `figures/final_*.png`, results to `code_full/final_results_101_205.mat`. The tuning experiments are in `code_full/tuning/` (`tune_report1.m`, `tune_vote.m`, `tune_refine.m`, `tune_gate.m`).
