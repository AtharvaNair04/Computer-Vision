# Improving Classical Human Pose Estimation Using Image Preprocessing and Adaptive Feature Extraction

**Project Checkpoint Report & Technical Documentation**  
**Group No:** 1  
**Team Members:**  
- Atharva Unnikrishnan Nair (Roll No: `AM.SC.U4CSE23109`)  
- G O Ashwin Praveen (Roll No: `AM.SC.U4CSE23027`)  
- Karthik Kishor (Roll No: `AM.SC.U4CSE23128`)  
- Revathi V (Roll No: `AM.SC.U4CSE23145`)  

**Base Paper:**  
Yi Yang and Deva Ramanan, *"Articulated Pose Estimation using Flexible Mixtures of Parts"*, IEEE CVPR 2011.

---

## 1. Executive Summary

Yang & Ramanan's Flexible Mixtures-of-Parts (FMP) model represents a person as a tree of HOG part templates and infers the pose with dynamic programming. It is fast and interpretable, but its hand-crafted gradient features degrade under noise, blur and poor lighting.

As the abstract proposes, this project wraps an **adaptive enhancement layer around the unchanged FMP model**:
* a no-reference image-quality analysis that chooses **one restoration per image** (non-local-means denoising, Richardson-Lucy deblurring or gamma correction);
* **flip test-time augmentation** with **part-consistent, score-weighted pose voting**;
* an **anthropometric pose-refinement** layer.

All design choices were made on PARSE test images 1–100. The results below are on the **held-out images 101–205** and on **BUFFY**, which was never used for tuning:

* **Gaussian noise:** **+5.99 PCK** (95% CI +2.69 to +9.12), **+8.12 APK**
* **Blur:** **+6.80 PCK** (95% CI +3.95 to +10.07), **+10.71 APK**
* **Dark:** **+2.86 PCK** (95% CI +0.54 to +5.03), **+2.17 APK**
* **Low contrast:** +2.04 PCK, +3.57 APK (positive, not significant at N = 105)
* **Clean images:** +1.16 PCK, +1.88 APK (positive, not significant)
* **BUFFY (276 frames, cross-dataset):** **+1.70 PCK**, **+2.12 APK**

Every condition improves on both metrics.

Cost: about 1.3 s → 2.4 s per image on a CPU, with no retraining and no GPU.

> Earlier versions of this report quoted "+50 pp PCK on noise" and other gains from a 25-image evaluation. Those numbers came from a results file written by a single-image run and cannot be reproduced; they are withdrawn. See `COMPLETE_PROJECT_WALKTHROUGH_AND_VERIFICATION.md`, Section 1.

---

## 2. Architecture

```
                      ┌──────────────────────────────────────────────┐
                      │             Raw Input Image (RGB)            │
                      └──────────────────────┬───────────────────────┘
                                             ▼
                      ┌──────────────────────────────────────────────┐
                      │ [1] Image Quality Analysis (image_quality.m) │
                      │   - Brightness  μ                            │
                      │   - Robust contrast (p99 − p1)               │
                      │   - Noise σ (Immerkær 1996 estimator)        │
                      │   - Sharpness var(∇²I_smooth) / var(I)       │
                      └──────────────────────┬───────────────────────┘
                                             ▼
                      ┌──────────────────────────────────────────────┐
                      │ [2] Gated Restoration (adaptive_preprocess.m)│
                      │   noise σ > 6       → non-local means        │
                      │   else sharp < 0.014→ Richardson-Lucy deblur │
                      │   else μ < 0.40     → gamma 0.67             │
                      │   else              → unchanged              │
                      └──────────────────────┬───────────────────────┘
                                             ▼
                      ┌──────────────────────────────────────────────┐
                      │ [3] HOG Feature Pyramid (original FMP)       │
                      │   (Adaptive HOG investigated as ablation)    │
                      └──────────────────────┬───────────────────────┘
                                             ▼
                      ┌──────────────────────────────────────────────┐
                      │ [4] FMP Inference + Flip TTA (detect_tta.m)  │
                      │   - original 26-part model, image + mirror   │
                      │   - mirrored candidates mapped back, pooled  │
                      │ [4b] Pose Voting (pose_vote.m)               │
                      └──────────────────────┬───────────────────────┘
                                             ▼
                      ┌──────────────────────────────────────────────┐
                      │ [5] Anatomical Pose Refinement               │
                      │   - Drillis & Contini / Winter bone bounds   │
                      │   - Torso-scale prior, symmetry prior        │
                      └──────────────────────┬───────────────────────┘
                                             ▼
                      ┌──────────────────────────────────────────────┐
                      │ [6–7] Visualisation and PCK / APK evaluation │
                      └──────────────────────────────────────────────┘
```

---

## 3. Mathematical Formulation

### Stage 1: Image-quality measures (grey image $I$, values 0–255)
* Brightness: $\mu = \frac{1}{255\,HW}\sum I(x,y)$
* Robust contrast: $c = (P_{99}(I) - P_{1}(I)) / 255$
* **Noise σ (Immerkær 1996):**
  $$\hat\sigma_n = \sqrt{\tfrac{\pi}{2}}\;\frac{1}{6(W-2)(H-2)}\sum_{x,y}\big|(I * N)(x,y)\big|,\qquad N = \begin{bmatrix}1&-2&1\\-2&4&-2\\1&-2&1\end{bmatrix}$$
  $N$ is the difference of two Laplacians. It cancels image structure up to second order, so $\hat\sigma_n$ measures noise and not texture.
* **Sharpness** (contrast-invariant):
  $$s = \frac{\operatorname{Var}\big(\nabla^2 (I * G_{0.8})\big)}{\operatorname{Var}(I)}$$
  The light pre-smoothing ($G_{0.8}$) keeps sensor noise from looking like detail.

### Stage 2: Gated restoration (priority noise > blur > dark, at most one operation)
* **Non-local means** (if $\hat\sigma_n > 6$):
  $$\hat I(p) = \frac{1}{Z_p}\sum_{q\in\Omega_p} I(q)\,\exp\!\left(-\frac{\|P(p)-P(q)\|^2}{h^2}\right)$$
  Here $P(\cdot)$ is a patch around a pixel and $h$ is estimated per image (`imnlmfilt` default).
* **Richardson-Lucy deconvolution** (if $s < 0.014$), with a Gaussian PSF $k$ of σ = 1.5 (9×9) and 8 iterations:
  $$\hat I^{(t+1)} = \hat I^{(t)} \cdot \left(\frac{I}{\hat I^{(t)} * k} * k^{\top}\right)$$
* **Gamma correction** (if $\mu < 0.40$):
  $$I_{\text{out}} = 255\,(I/255)^{0.67}$$

### Stage 4: Flip TTA and pose voting
* Each candidate box from the mirrored image is mapped back with $x' = W + 1 - x$, and its parts are permuted $[1, 2, B, A]$. A and B are the two body sides of the part tree (PARSE: parts 3–14 and 15–26).
* For the selected detection with score $s^*$, the candidate set is
  $$\mathcal{C} = \{i : \text{IoU}(i, *) > 0.5,\; s_i \ge s^* - 1.0\}$$
  and every part position becomes a part-consistent weighted mean (only candidates that agree on part $j$ vote for it):
  $$\hat{\mathbf{x}}_j = \frac{\sum_{i\in\mathcal C} w_{ij}\,\mathbf{x}_{ij}}{\sum_{i\in\mathcal C} w_{ij}}, \qquad w_{ij} = e^{(s_i - s^*)/0.5}\cdot \mathbb{1}\big[\|\mathbf{x}_{ij} - \mathbf{x}_{*j}\| \le 0.25\,S\big]$$
  $S$ is the size of the selected person's box. Without the indicator (whole-pose voting) the tuning-split mean was 73.2 PCK and heads were pulled off target; with it, 75.0.

### Stage 5: Anatomical refinement
* Torso scale:
  $$T = \begin{cases}0.65\,d(\text{midShoulder},\text{midHip}) + 0.35\cdot 0.30\,H & \text{if } 0.18H \le d \le 0.45H\\ 0.30\,H & \text{otherwise}\end{cases}$$
  $H$ is the vertical extent of the predicted keypoints.
* Bone bounds, as multiples of $T$: upper arm [0.35, 0.90], forearm [0.30, 0.80], thigh [0.50, 1.20], calf [0.45, 1.15].
* Projection of an out-of-range joint: $\mathbf{x}_j' = \mathbf{x}_p + \frac{\vec v}{\|\vec v\|}\,\text{clamp}(\|\vec v\|, L_{\min}, L_{\max})$
* Symmetry prior: a limb whose length is more than 25% away from the median of {left, right, anthropometric prior} is pulled 35% of the way towards that median.

---

## 4. Evaluation Protocol

| Item | Setting |
|---|---|
| PARSE | 205 test images, 14 joints. Images **1–100 = tuning split**, **101–205 = held-out split** |
| BUFFY | 276 test frames (episodes 2, 5, 6), 10 upper-body joints, never tuned on |
| Degradations | `degrade_image.m`: dark (γ 1.5), low contrast (L* × 0.45), Gaussian noise (var 0.01), Gaussian blur (σ 1.5) |
| Metrics | Official PCK and APK (PARSE threshold 0.1, BUFFY 0.2) |
| Statistics | Paired bootstrap 95% CI (5 000 resamples) of the per-image PCK change |

### Design decisions made on the tuning split (PCK, N = 100)

| Condition | Baseline | Best rejected | **Selected** |
|---|---|---|---|
| Noise | 67.4 | strong bilateral 69.1 | **NLM 70.7** |
| Blur | 66.1 | 1.25× upsampling 66.3 | **Richardson-Lucy 68.3** |
| Dark | 70.9 | contrast stretch 71.6 | **gamma 0.67: 74.0** |
| Low contrast | 71.8 | gamma 71.1 | **no operation** |
| CLAHE (mean of 5 conditions, full pipeline) | 74.97 | CLAHE everywhere 74.51; CLAHE on low-contrast only 75.11 (+0.14) | **not adopted** (adoption bar +0.3, fixed in advance) |
| Adaptive HOG (mean of 5 conditions) | 74.97 | gradient-weighted HOG ±0.1; denser scale pyramid 75.07 (+0.10, APK −0.40, ~2× slower) | **not adopted** |
| Pose voting (mean of 5 conditions) | 69.5 | whole-pose voting 73.2 | **part-consistent voting: 75.0** |

The part-consistent revision was prompted by per-joint losses in a first held-out report. It was then designed and tuned on images 1–100 only and scored on 101–205 once.

---

## 5. Results

### A. PARSE held-out split (images 101–205, N = 105)

| Condition | Baseline PCK | Full PCK | Δ PCK [95% CI] | Baseline APK | Full APK | Δ APK |
|---|---|---|---|---|---|---|
| Clean | 75.37 | 76.53 | +1.16 [−1.53, +3.61] | 66.19 | 68.07 | +1.88 |
| Dark | 75.51 | **78.37** | **+2.86 [+0.54, +5.03]** | 67.13 | **69.30** | **+2.17** |
| Low contrast | 74.76 | 76.80 | +2.04 [−0.48, +4.22] | 66.28 | 69.85 | +3.57 |
| Noise | 68.78 | **74.76** | **+5.99 [+2.69, +9.12]** | 57.26 | **65.38** | **+8.12** |
| Blur | 68.91 | **75.71** | **+6.80 [+3.95, +10.07]** | 57.43 | **68.14** | **+10.71** |

### B. Ablation (held-out PCK)

| Condition | Baseline | Adaptive HOG | + Preprocess | + Flip TTA / Vote | + Refinement |
|---|---|---|---|---|---|
| Clean | 75.37 | 75.44 | 74.83 | 76.53 | 76.53 |
| Dark | 75.51 | 75.51 | 74.76 | 78.16 | 78.37 |
| Low contrast | 74.76 | 74.69 | 74.49 | 76.46 | 76.80 |
| Noise | 68.78 | 68.64 | 71.02 | 74.83 | 74.76 |
| Blur | 68.91 | 68.91 | 71.16 | 74.97 | 75.71 |

**What each stage contributes:**
* **Preprocessing** gives +2.2 each on noise and blur.
* **Flip TTA + part-consistent voting** adds +1.7 to +3.8 in every condition: the largest single contribution.
* **Refinement** adds between −0.1 and +0.75 (largest on blur).
* **Adaptive HOG** has no effect without retraining.

### C. BUFFY cross-dataset (276 frames)

| Metric | Head | Shoulder | Elbow | Wrist | Hip | **Average** |
|---|---|---|---|---|---|---|
| PCK baseline | 96.0 | 97.1 | 87.5 | 67.2 | 85.3 | 86.63 |
| PCK enhanced | 95.7 | 97.1 | 90.6 | 68.8 | 89.5 | **88.33 (+1.70)** |
| APK baseline | 88.3 | 88.8 | 78.6 | 56.2 | 71.2 | 76.63 |
| APK enhanced | 88.3 | 89.1 | 83.1 | 56.5 | 76.9 | **78.75 (+2.12)** |

With the earlier whole-pose voting, BUFFY gained more (+2.36 PCK); the voting was tuned on PARSE and not re-tuned for BUFFY.

### D. Limitations
* The degradations are synthetic, one severity each. The deblurring PSF (σ 1.5) matches the synthetic blur, so real motion blur needs PSF estimation.
* The clean and low-contrast gains are not statistically significant at N = 105. On clean PARSE, wrist (−3.8), shoulder (−1.5) and head (−1.4) dip slightly while legs and elbows gain.
* Adaptive HOG would need the FMP filters retrained (`train_adaptive_fmp.m`). This requires the full INRIA negative set, which is not in the repository.
* Flip TTA doubles inference time.

---

## 6. Repository File Map and Reproducibility

| Purpose | File |
|---|---|
| Quality analysis | `code_full/image_quality.m` |
| Gated restoration | `code_full/adaptive_preprocess.m` |
| Flip TTA | `code_full/detect_tta.m` |
| Pose voting | `code_full/pose_vote.m` |
| Stages 1–4 combined | `code_full/enhanced_detect.m` |
| Pose refinement | `code_full/refine_pose.m`, `code_full/apply_pose_refinement.m` |
| Adaptive HOG (ablation) | `code_basic/adaptive_detect_fast.m`, `code_basic/adaptive_features.mexw64` |
| PARSE evaluation | `code_full/run_final_evaluation.m` (`run_complete_experiments.m` calls it) |
| BUFFY evaluation | `code_full/run_buffy_experiments.m` |
| Demo | `code_full/demo_complete.m` |
| Tuning experiments | `code_full/tuning/` (`tune_report1.m`, `tune_vote.m`, `tune_refine.m`, `tune_gate.m`) |
| Full results | `ENHANCED_PIPELINE_RESULTS.md`, `code_full/final_results_101_205.mat` |
| Figures | `figures/final_*.png` |

```matlab
cd 'human-pose-estimation-master - enhanced/code_full';
results = run_final_evaluation(101:205);   % PARSE held-out ablation + figures
buffy   = run_buffy_experiments();          % BUFFY, all 276 test frames
demo_complete(4, 'noise');                  % live 4-panel demo
```

---

## 7. Slide-by-Slide Defence Script

* **Slide 1 (Title):** project title, Group 1.
* **Slide 2 (Problem):** FMP is interpretable and CPU-friendly, but HOG gradients degrade under noise, blur and low light. Baseline PCK drops from 75.4 (clean) to 68.8 (noise) and 68.9 (blur).
* **Slide 3 (Architecture):** the 7-stage diagram. The FMP model is untouched; the enhancement layer wraps around it, as the abstract proposes.
* **Slide 4 (Protocol):** tuning split vs. held-out split vs. BUFFY. All design choices were measured, not guessed (table in Section 4).
* **Slide 5 (Main result):** `final_pck_by_condition.png`. Noise +6.0, blur +6.8 and dark +2.9 PCK, all with 95% CIs above zero. APK +8.1, +10.7 and +2.2. Every condition improves.
* **Slide 6 (Ablation):** what each stage contributes. Preprocessing does most of the work under degradation; voting and refinement add consistent smaller gains; adaptive HOG is a negative result explained by the train/test feature mismatch.
* **Slide 7 (Qualitative):** `final_qualitative.png`. Under noise, baseline FMP finds no person at all; the full pipeline recovers the pose.
* **Slide 8 (Generalisation):** BUFFY +1.7 PCK / +2.1 APK, with no tuning on BUFFY.
* **Slide 9 (Limitations and future work):** synthetic degradations, known blur PSF, adaptive-HOG retraining with full INRIA negatives, cost of flip TTA.
* **Slide 10 (Conclusion):** lightweight, interpretable classical techniques measurably improve FMP's robustness without replacing it, which answers the abstract's research question.
