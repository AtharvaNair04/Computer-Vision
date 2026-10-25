# Project Walkthrough, Technical Rationale, Verified Results and Defence Guide

**Project Title:** Improving Classical Human Pose Estimation Using Image Preprocessing and Adaptive Feature Extraction  
**Team Details:** Group No: 1  
- **Atharva Unnikrishnan Nair** (Roll No: `AM.SC.U4CSE23109`)  
- **G O Ashwin Praveen** (Roll No: `AM.SC.U4CSE23027`)  
- **Karthik Kishor** (Roll No: `AM.SC.U4CSE23128`)  
- **Revathi V** (Roll No: `AM.SC.U4CSE23145`)  

**Base Paper:**  
Yi Yang and Deva Ramanan, *"Articulated Pose Estimation using Flexible Mixtures of Parts"*, IEEE CVPR 2011.  
Official Base Repo: `https://github.com/yangyi02/human-pose-estimation`  
Datasets: **PARSE** and **BUFFY** (INRIA Person is the negative set used for training)

> All numbers in this document come from `code_full/run_final_evaluation.m` and `code_full/run_buffy_experiments.m` and can be reproduced exactly (Section 6). FMP inference is deterministic, so re-running gives identical numbers. The detailed results are in [`ENHANCED_PIPELINE_RESULTS.md`](ENHANCED_PIPELINE_RESULTS.md).

---

## 1. Development History

The project went through three versions. The first two are kept in git history.

### Version 1: mid-review
* In `adaptive_preprocess_v2.m`, noise and blur were detected but not acted on (`'High noise detected - unchanged'`). Only mild gamma correction and min-max contrast stretching were active.
* Adaptive HOG, pose refinement and BUFFY evaluation were listed as future work.
* The metrics were flat or slightly negative: clean −0.28 pp PCK, blur −0.10 pp APK.

### Version 2: hand-calibrated preprocessing
* Added a bilateral filter for noise, unsharp masking for blur, a dark-image gamma, and the anatomical pose refinement module.
* It was evaluated on only 25 PARSE images. Its parameters were chosen while looking at those same test images.
* **Its reported numbers cannot be reproduced.** The widely quoted "+50 pp PCK on noise" (14.29% → 64.29%) came from a results file written by a run on a **single image**. Every per-joint value in it is 0, 50 or 100. Re-run on 25 images, version 2 scored **below** the baseline on noise (−0.86 PCK), blur (−0.86) and low contrast (−0.86) and clean images (−0.57). Only dark images improved (+1.71).

### Version 3: validated pipeline (current)
* **Proper protocol:** PARSE test images **1–100** were used for every design decision. Images **101–205** were held out and only used for the final report. BUFFY was never used for tuning.
* **Systematic search:** 15 image transforms × 5 conditions × 205 images were cached, and each preprocessing choice was decided by measured PCK rather than intuition.
* **New stages:** a better noise estimator, NLM denoising, Richardson-Lucy deblurring, flip test-time augmentation and pose voting.
* **Significance testing:** gains are reported with paired-bootstrap 95% confidence intervals.

---

## 2. Technical Rationale (the "why")

### Finding 1: HOG already normalises contrast, so contrast enhancement does not help
FMP's HOG features normalise each 2×2-cell block: $\mathbf{f} = \mathbf{v} / \sqrt{\|\mathbf{v}\|^2 + \epsilon}$. A global contrast change scales all gradients in a block equally and is therefore removed by this normalisation.

Tested on the tuning split (low-contrast images, PCK): baseline 71.8, contrast stretch 70.2, gamma 71.1. **Conclusion:** low-contrast images are best left untouched. This is also why CLAHE from the original proposal was dropped.

### Finding 2: the noise estimator mattered as much as the denoiser
Version 2 measured noise as std(I − Gauss(I, σ=1)). That measure responds to image texture as well as noise, so it fired on clean images and missed some noisy ones (it triggered on 18/25 noisy images).

Version 3 uses **Immerkær's fast noise estimator** (1996). It convolves with a mask that cancels image structure up to second order, so it measures noise only. On all 205 PARSE images:
* clean images: σ ≤ 9.2
* noisy images: σ ≥ 9.9

The two groups separate cleanly.

### Finding 3: non-local means suits HOG better than a bilateral filter
At the test noise level (σ ≈ 17 grey levels), a bilateral filter with σ_r = 20 either leaves grain or, with larger windows, smears thin limbs. Non-local means averages pixels from *similar patches* anywhere in a search window. It removes grain while keeping limb edges, which HOG's orientation bins depend on.

Tuning split, noise condition, PCK:

| Method | PCK |
|---|---|
| Baseline | 67.4 |
| Version 2 bilateral | 67.4 |
| Median 3×3 | 68.3 |
| Wiener 5×5 | 68.8 |
| Gaussian σ 1 | 69.0 |
| Strong bilateral | 69.1 |
| **NLM** | **70.7** |

### Finding 4: blur needs deconvolution, not sharpening
Unsharp masking boosts edge contrast but cannot restore detail that the blur removed (tuning PCK: 65.9 vs baseline 66.1). Richardson-Lucy deconvolution inverts the blur model (Gaussian PSF, 8 iterations) and gives **68.3**.

It hurts sharp images (69.0 vs 71.4 on clean), so it must be gated. The gate uses a **contrast-normalised Laplacian energy**, var(∇²I)/var(I), computed after light smoothing. This keeps low-contrast or noisy images from being mistaken for blurred ones.

### Finding 5: FMP's single best configuration is noisy, so averaging helps
FMP returns the highest-scoring pose. Many near-identical candidates with almost the same score exist, shifted by one HOG cell or using a different part mixture. Two additions exploit this:
* **Flip test-time augmentation:** run the unchanged FMP on the mirrored image too, map those detections back (reflect x, swap left/right parts) and pool both sets.
* **Pose voting:** replace each output joint by the score-weighted mean over candidates of the same person (IoU > 0.7, score within 0.25, weight $e^{(s_i - s^*)/0.1}$).

Under degradation these two stages add **+1.6 to +2.0 PCK** beyond preprocessing (held-out noise and blur). On held-out clean images they gave no gain.

### Finding 6: anatomical refinement gives a small, consistent gain
On top of voting, the bone-length bounds and symmetry prior add +0.1 to +0.7 PCK in every held-out condition (+0.26 on average on the tuning split). The robust torso-scale prior (0.30 × the vertical extent of the predicted keypoints) keeps one misdetected shoulder or hip from rescaling every limb.

### Finding 7: Adaptive HOG does not help without retraining
`adaptive_features` changes the feature distribution, but the FMP filters were trained on the original features. On every condition, accuracy stays within ±0.1 of the baseline.

Retraining (`train_adaptive_fmp.m`) needs the full INRIA negative set (frames 615–1832), which is not in the repository. This remains the main open item.

---

## 3. Final Pipeline (what is implemented)

| Stage (abstract diagram) | File | Implementation |
|---|---|---|
| 1. Image-quality analysis | `code_full/image_quality.m` | brightness, robust contrast (1–99th percentile), Immerkær noise σ, contrast-normalised Laplacian sharpness |
| 2. Adaptive preprocessing | `code_full/adaptive_preprocess.m` | at most one operation per image, priority noise > blur > dark:<br>• σ > 6 → NLM (`imnlmfilt`)<br>• sharpness < 0.014 → Richardson-Lucy (σ 1.5, 8 iterations)<br>• brightness < 0.40 → gamma 0.67<br>• otherwise unchanged |
| 3. HOG pyramid | `code_full/detection/featpyramid.m` | original, unchanged |
| 3b. Adaptive HOG (investigated) | `code_basic/adaptive_detect_fast.m` | reported as an ablation row only |
| 4. FMP inference + flip TTA | `code_full/detect_tta.m` | original `detect_fast` on the image and its mirror; part mirror map `[1 2, side B, side A]` |
| 4b. Pose voting | `code_full/pose_vote.m` | score-weighted keypoint averaging |
| 5. Pose refinement | `code_full/refine_pose.m`, `apply_pose_refinement.m` | anthropometric bounds (Drillis & Contini 1966; Winter 2009) and symmetry prior |
| 6–7. Visualisation / evaluation | `demo_complete.m`, `run_final_evaluation.m`, `run_buffy_experiments.m` | demo, ablation, bootstrap CIs, figures |

`code_full/enhanced_detect.m` chains stages 1–4. The tuning experiments are in `code_full/tuning/`.

---

## 4. Verified Results

### A. PARSE held-out images 101–205 (N = 105), PCK %

| Condition | Baseline FMP | Adaptive HOG | + Preprocess | + Flip TTA / Vote | **Full** | Full − Baseline [95% CI] |
|---|---|---|---|---|---|---|
| Clean | 75.37 | 75.44 | 74.83 | 74.42 | **75.10** | −0.27 [−2.93, +1.97] |
| Dark | 75.51 | 75.51 | 74.76 | 76.94 | **77.28** | +1.77 [−0.65, +4.15] |
| Low contrast | 74.76 | 74.69 | 74.49 | 74.69 | **75.10** | +0.34 [−1.90, +2.31] |
| Noise | 68.78 | 68.64 | 71.02 | 72.65 | **72.79** | **+4.01 [+0.75, +7.21]** |
| Blur | 68.91 | 68.91 | 71.16 | 73.13 | **73.40** | **+4.49 [+1.77, +7.69]** |

### B. PARSE held-out images 101–205, APK %

| Condition | Baseline | Full | Δ |
|---|---|---|---|
| Clean | 66.19 | 66.28 | +0.09 |
| Dark | 67.13 | 68.24 | +1.11 |
| Low contrast | 66.28 | 67.37 | +1.09 |
| Noise | 57.26 | 63.10 | **+5.84** |
| Blur | 57.43 | 65.30 | **+7.87** |

### C. Per-joint PCK, clean held-out images (Baseline → Full)

| Head | Shoulder | Elbow | Wrist | Hip | Knee | Ankle |
|---|---|---|---|---|---|---|
| 91.9 → 88.6 | 88.6 → 88.6 | 68.6 → **72.4** | 51.4 → 44.8 | 80.0 → **81.4** | 77.6 → 77.1 | 69.5 → **72.9** |

On clean images the pipeline trades accuracy between joints (elbow and ankle up; head and wrist down), so the net change is zero.

### D. BUFFY cross-dataset (all 276 test frames, no tuning on BUFFY)

| Metric | Head | Shoulder | Elbow | Wrist | Hip | **Average** |
|---|---|---|---|---|---|---|
| PCK baseline | 96.0 | 97.1 | 87.5 | 67.2 | 85.3 | 86.63 |
| PCK enhanced | 96.2 | 97.1 | 90.0 | 71.4 | 90.2 | **88.99 (+2.36)** |
| APK baseline | 88.3 | 88.8 | 78.8 | 56.7 | 71.5 | 76.82 |
| APK enhanced | 88.5 | 89.1 | 82.0 | 59.8 | 77.5 | **79.40 (+2.59)** |

BUFFY uses preprocessing + flip TTA + voting. Refinement is defined for PARSE's 14 joints, so it is not applied here.

### E. Runtime (single MATLAB process, `demo_complete`)

| Stage | Time per image |
|---|---|
| Quality analysis + restoration | ≈ 0.13 s (NLM is the most expensive operation) |
| FMP inference, one pass | ≈ 1.2–1.3 s |
| FMP inference with flip TTA (two passes) | ≈ 2.3–2.4 s |
| Voting + refinement | < 0.01 s |
| **Baseline total / Enhanced total** | **≈ 1.3 s / ≈ 2.4 s** |

The flip pass doubles inference time. Preprocessing on its own is nearly free and still gives +2.2 PCK on noise and blur.

### F. Summary of what the evidence supports
* **Significant (CI excludes 0):** noise +4.0 PCK / +5.8 APK; blur +4.5 PCK / +7.9 APK.
* **Positive, not significant at N = 105:** dark +1.8 PCK; low contrast +0.3 PCK / +1.1 APK.
* **No regression:** clean −0.3 PCK / +0.1 APK.
* **Generalises:** BUFFY +2.4 PCK / +2.6 APK without any tuning there.

---

## 5. Likely Viva Questions and Answers

**Q1. How do you know you did not overfit the test set?**  
Every threshold and operator was chosen on PARSE test images 1–100. The headline numbers are on images 101–205, which were never used for any decision, plus BUFFY, which was never tuned on. On the clean condition, flip TTA looked like +3 PCK on the tuning split but gave nothing on the held-out split. We report that honestly, and it shows the protocol works.

**Q2. Your degradations are synthetic. Would this work on real images?**  
Partly. The noise and darkness gates use generic statistics (Immerkær σ, mean luminance), and BUFFY consists of real TV frames where the pipeline gained +2.4 PCK. The deblurring assumes a Gaussian PSF of σ 1.5, which matches our synthetic blur. Real motion blur would need PSF estimation, for example blind deconvolution. That is a stated limitation.

**Q3. Why did adaptive HOG not help, when it was in the proposal?**  
FMP's part filters are linear classifiers trained on the original HOG distribution. Changing the features at test time without retraining moves them away from what the filters expect, so accuracy stays at the baseline (±0.1). The retraining code exists (`train_adaptive_fmp.m`), but it needs the full INRIA negative set. We report adaptive HOG as an investigated and negative result.

**Q4. Is a 4-point gain meaningful?**  
For a fixed 2011 model with no retraining, yes. It recovers about 60% of the drop that noise causes (clean 75.4 → noisy 68.8 → enhanced 72.8) and about 70% for blur. The bootstrap intervals exclude zero.

**Q5. What does flip TTA cost, and is it still "lightweight"?**  
It doubles FMP inference (≈ 1.3 s → 2.4 s per image on CPU) but needs no GPU and no new training. If cost matters, preprocessing alone keeps about half of the gain at almost no extra cost.

---

## 6. How to Run Everything

```matlab
cd 'c:\Users\athar\Desktop\Computer-Vision\human-pose-estimation-master - enhanced\code_full'

% PARSE ablation on the held-out split (cached per image in cache/final/)
results = run_final_evaluation(101:205);     % or run_complete_experiments()

% Faster: run detection in 5 MATLAB windows in parallel, then the metrics
%   window k (k = 1..5):  run_final_evaluation(1:205, 'shard', [k 5])
%   afterwards:           run_final_evaluation(101:205)

% BUFFY cross-dataset benchmark (all 276 test frames)
buffy = run_buffy_experiments();

% Live demo (4 panels: input, baseline, enhanced, refined)
demo_complete(4, 'noise');
demo_complete(1, 'blur');
demo_complete(1, 'dark');
```

Figures produced (in `figures/`):
* `final_pck_by_condition.png`, `final_apk_by_condition.png`: ablation bar charts
* `final_per_joint_clean.png`: per-joint PCK on clean images
* `final_pck_curves.png`: PCK vs. threshold for each condition
* `final_qualitative.png`: the largest per-condition improvement, baseline vs. full pipeline

The older `robustness_comparison_*.png`, `keypoint_breakdown_clean.png`, `pck_threshold_curves.png` and `qualitative_comparison_*.png` come from the 25-image version-2 evaluation and are superseded.

---

## 7. Prompt for Independent Review

Paste the block below into another model or give it to a reviewer to check the work:

```text
You are a computer vision reviewer. Evaluate this undergraduate project that enhances the classical
Flexible Mixtures-of-Parts (FMP) pose estimator (Yang & Ramanan, CVPR 2011) without retraining it.

Pipeline: (1) no-reference quality analysis: Immerkaer noise sigma, mean brightness, robust contrast,
contrast-normalised Laplacian sharpness; (2) one gated restoration per image: noise sigma > 6 ->
non-local-means; sharpness < 0.014 -> Richardson-Lucy deconvolution (Gaussian PSF sigma 1.5); brightness
< 0.40 -> gamma 0.67; (3-4) unchanged HOG + FMP inference on the image and its mirror, detections pooled;
(4b) score-weighted pose voting over near-duplicate candidates (IoU > 0.7, score within 0.25, T = 0.1);
(5) anthropometric bone-length and symmetry refinement.

Protocol: PARSE test images 1-100 used for all tuning; 101-205 held out; BUFFY (276 frames) not tuned on.
Degradations are synthetic: dark (gamma 1.5), low contrast, Gaussian noise (var 0.01), Gaussian blur (sigma 1.5).

Held-out PARSE PCK (baseline -> full, 95% bootstrap CI of change): clean 75.37 -> 75.10 [-2.93, +1.97];
dark 75.51 -> 77.28 [-0.65, +4.15]; low contrast 74.76 -> 75.10 [-1.90, +2.31];
noise 68.78 -> 72.79 [+0.75, +7.21]; blur 68.91 -> 73.40 [+1.77, +7.69].
Held-out APK: noise +5.84, blur +7.87, dark +1.11, low contrast +1.09, clean +0.09.
BUFFY: PCK 86.63 -> 88.99, APK 76.82 -> 79.40. Runtime ~1.3 s -> ~2.4 s per image (flip doubles inference).
Adaptive HOG features without retraining: no change (+/-0.1).

Please assess: soundness of the protocol, whether the conclusions follow from the numbers, the main
threats to validity, and what further experiment would most strengthen the work.
```
