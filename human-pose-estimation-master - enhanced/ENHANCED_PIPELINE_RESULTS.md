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
| **Held-out split** | PARSE test images **101–205**: never used to make a choice. **These are the headline numbers.** Disclosure: the part-consistent voting revision was prompted by per-joint losses in a first held-out report; it was then designed and tuned on 1–100 only and scored here once. |
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
| 4b. Pose voting | `pose_vote.m` | **Part-consistent, score-weighted voting**: candidates of the same person (IoU > 0.5, score within 1.0 of the best, weight exp(Δs/0.5)) vote for a part only if their part lies within 0.25 × body size of the selected detection's part |
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
| CLAHE (abstract stage 2) | full pipeline 74.97 (mean of 5 conditions) | CLAHE on every unprocessed image 74.51 (clean −1.8); CLAHE only on low-contrast images 75.11 (+0.14, APK −0.04) | not adopted: below the +0.3 adoption bar fixed in advance (`tune_abstract.m`) |
| Adaptive HOG (abstract stage 3) | — | gradient-weighted HOG (`adaptive_features`): ±0.1 in every condition; denser scale pyramid (15 instead of 10 levels per octave): +0.10 PCK, −0.40 APK, about 2× slower | not adopted. Changing cell size, block size or orientation bins changes the feature dimensions the FMP filters were trained on, so it needs retraining |
| Pose voting | — | whole-pose voting (no per-part check): mean 73.2, pulled heads off target (head −4.5 under blur) | **part-consistent voting: mean 75.0** (`tune_vote2.m`) |
| All | — | pooling the unprocessed image as well (2× cost) gave only +0.1 | not used |

---

## 3. Results on the held-out PARSE images (101–205, N = 105)

### PCK (%)

| Condition | Baseline FMP | Adaptive HOG | + Preprocess | + Flip TTA / Vote | **+ Refinement (full)** | Full − Baseline [95% CI] |
|---|---|---|---|---|---|---|
| Clean | 75.37 | 75.44 | 74.83 | 76.53 | **76.53** | +1.16 [−1.53, +3.61] |
| Dark | 75.51 | 75.51 | 74.76 | 78.16 | **78.37** | **+2.86 [+0.54, +5.03]** |
| Low contrast | 74.76 | 74.69 | 74.49 | 76.46 | **76.80** | +2.04 [−0.48, +4.22] |
| Noise | 68.78 | 68.64 | 71.02 | 74.83 | **74.76** | **+5.99 [+2.69, +9.12]** |
| Blur | 68.91 | 68.91 | 71.16 | 74.97 | **75.71** | **+6.80 [+3.95, +10.07]** |

### APK (%)

| Condition | Baseline FMP | Adaptive HOG | + Preprocess | + Flip TTA / Vote | **Full** | Δ |
|---|---|---|---|---|---|---|
| Clean | 66.19 | 66.28 | 66.28 | 68.12 | **68.07** | +1.88 |
| Dark | 67.13 | 67.14 | 66.38 | 69.06 | **69.30** | +2.17 |
| Low contrast | 66.28 | 66.65 | 65.98 | 69.51 | **69.85** | +3.57 |
| Noise | 57.26 | 57.14 | 60.41 | 65.46 | **65.38** | **+8.12** |
| Blur | 57.43 | 57.44 | 61.01 | 67.28 | **68.14** | **+10.71** |

### Per-joint PCK, clean held-out images (Baseline → Full)

| Head | Shoulder | Elbow | Wrist | Hip | Knee | Ankle |
|---|---|---|---|---|---|---|
| 91.9 → 90.5 | 88.6 → 87.1 | 68.6 → **70.5** | 51.4 → 47.6 | 80.0 → **83.3** | 77.6 → **81.4** | 69.5 → **75.2** |

### Full PARSE test set (1–205, N = 205), for reference only (half of it is the tuning split)

| Condition | Baseline PCK | Full PCK | Δ PCK [95% CI] | Baseline APK | Full APK |
|---|---|---|---|---|---|
| Clean | 73.45 | 76.55 | +3.10 [+1.29, +4.88] | 64.48 | 68.60 |
| Dark | 73.24 | 77.18 | +3.94 [+2.26, +5.75] | 64.76 | 68.79 |
| Low contrast | 73.31 | 75.75 | +2.44 [+0.87, +3.90] | 64.97 | 68.90 |
| Noise | 68.08 | 74.18 | +6.10 [+3.94, +8.26] | 57.52 | 65.66 |
| Blur | 67.56 | 74.95 | +7.39 [+5.19, +9.62] | 57.21 | 67.59 |

## 4. BUFFY cross-dataset results (all 276 test frames, no BUFFY tuning)

Pipeline: preprocessing + flip TTA + part-consistent pose voting. Refinement is not used here because it is defined for PARSE's 14 joints. The baseline keeps each frame's 2 000 best candidates (`run_buffy_experiments.m` caches them), which moves the baseline APK from 76.82 to 76.63.

| Metric | Head | Shoulder | Elbow | Wrist | Hip | **Average** |
|---|---|---|---|---|---|---|
| PCK baseline | 96.0 | 97.1 | 87.5 | 67.2 | 85.3 | 86.63 |
| PCK enhanced | 95.7 | 97.1 | 90.6 | 68.8 | 89.5 | **88.33 (+1.70)** |
| APK baseline | 88.3 | 88.8 | 78.6 | 56.2 | 71.2 | 76.63 |
| APK enhanced | 88.3 | 89.1 | 83.1 | 56.5 | 76.9 | **78.75 (+2.12)** |

With the earlier whole-pose voting, BUFFY gained more (+2.36 PCK / +2.59 APK) while PARSE gained less. The voting parameters were chosen on PARSE only, and we did not re-tune them for BUFFY.

## 5. Cost

Single-process timing (`demo_complete`): baseline FMP ≈ 1.3 s per image, full pipeline ≈ 2.4 s. The flip pass doubles FMP inference. Quality analysis plus restoration adds about 0.13 s. Refinement and voting are negligible.

---

## 6. Honest reading of the results

* **Held-out gains with a CI that excludes 0:**
  * **Noise:** +6.0 PCK / +8.1 APK
  * **Blur:** +6.8 PCK / +10.7 APK
  * **Dark:** +2.9 PCK / +2.2 APK

  Noisy and blurred images end up within 0.6 PCK of the clean baseline (75.4).
* **Positive but not significant on 105 images:** low contrast (+2.0 PCK, +3.6 APK) and clean (+1.2 PCK, +1.9 APK). Every condition improves on both metrics.
* **Per joint:** on clean PARSE, legs and elbows gain (ankle +5.7, knee +3.8, hip +3.3). Wrist (−3.8), shoulder (−1.5) and head (−1.4) dip slightly, but **no joint is significantly worse**. Paired-bootstrap 95% CIs of the per-joint change:

  | Head | Shoulder | Elbow | Wrist | Hip | Knee | Ankle |
  |---|---|---|---|---|---|---|
  | [−5.2, +1.9] | [−5.7, +2.4] | [−2.9, +6.7] | [−9.5, +1.9] | [−1.0, +8.1] | [−0.5, +8.1] | **[+1.4, +10.5]** |

  We did not tune further on the held-out images to remove the dips; that would be fitting the test set.
* **Cross-dataset:** +1.7 PCK / +2.1 APK on BUFFY, with no tuning there. BUFFY head is −0.4, i.e. one joint in 276 frames.
* **Adaptive HOG** (`adaptive_features`) does not change accuracy. Without retraining the FMP filters on the new feature distribution, it cannot. Retraining (`train_adaptive_fmp.m`) needs the full INRIA negative set (frames 615–1832), which is not in this repository.

### Limitations

* The degradations are synthetic and at one severity each. The deblurring PSF (σ 1.5) matches the synthetic blur, so real motion blur would need PSF estimation.
* The thresholds come from 100 PARSE images. BUFFY shows they transfer, but a real-image test is still missing.
* The 2× inference cost comes from the flip pass. Preprocessing alone (no TTA) is almost free and still gives +2.2 PCK on noise and blur.

### Corrections to earlier project reports

The mid-review-era claim of "+50 pp on noise" came from a `complete_evaluation_results.mat` written by a run on a **single image** (every per-joint value is 0, 50 or 100). Use the tables in this document instead.

---

## 7. How to reproduce

```matlab
cd 'human-pose-estimation-master - enhanced/code_full'

% PARSE, held-out split (about 45 min single-process; cached in cache/final/)
results = run_final_evaluation(101:205);

% Optional: shard detection over several MATLAB processes first, e.g. 5 of them:
%   run_final_evaluation(1:205, 'shard', [k 5])   % k = 1..5, one per process
% then call run_final_evaluation(...) once more for the metrics.

% BUFFY, all 276 test frames (shardable the same way: run_buffy_experiments([], [k 5]))
buffy = run_buffy_experiments();

% Interactive demo
demo_complete(4, 'noise');
demo_complete(1, 'blur');
```

Figures are written to `figures/final_*.png`, results to `code_full/final_results_101_205.mat`. The tuning experiments are in `code_full/tuning/` (`tune_report1.m`, `tune_vote.m`, `tune_vote2.m`, `tune_joints.m`, `tune_refine.m`, `tune_gate.m`).
