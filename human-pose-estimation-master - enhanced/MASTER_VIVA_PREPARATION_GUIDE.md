# MASTER VIVA PREPARATION GUIDE: END-TO-END TECHNICAL MANUAL & DEFENCE
## Improving Classical Human Pose Estimation Using Image Preprocessing and Adaptive Feature Extraction

---

### GROUP & PROJECT PROFILE
* **Group No:** 1
* **Team Members:**
  * **Atharva Unnikrishnan Nair** (Roll No: `AM.SC.U4CSE23109`)
  * **G O Ashwin Praveen** (Roll No: `AM.SC.U4CSE23027`)
  * **Karthik Kishor** (Roll No: `AM.SC.U4CSE23128`)
  * **Revathi V** (Roll No: `AM.SC.U4CSE23145`)
* **Base Paper:** Yi Yang and Deva Ramanan, *"Articulated Pose Estimation using Flexible Mixtures of Parts"*, IEEE CVPR 2011 / IEEE TPAMI 2013.
* **Benchmarks:** PARSE Dataset (205 images), BUFFY Dataset (276 frames), INRIA Person Background Dataset.

---

# TABLE OF CONTENTS
1. [The 60-Second Elevator Pitch & Core Motivation](#1-the-60-second-elevator-pitch--core-motivation)
2. [Base Paper Theory: Yang & Ramanan FMP Model](#2-base-paper-theory-yang--ramanan-fmp-model)
3. [End-to-End System Architecture](#3-end-to-end-system-architecture)
4. [Line-by-Line Code & Module Walkthrough](#4-line-by-line-code--module-walkthrough)
   - [Module 1: Image Quality Assessment (`image_quality.m`)](#module-1-image-quality-assessment-image_qualitym)
   - [Module 2: Gated Adaptive Restoration (`adaptive_preprocess.m`)](#module-2-gated-adaptive-restoration-adaptive_preprocessm)
   - [Module 3: Feature Extraction & The Adaptive HOG Study](#module-3-feature-extraction--the-adaptive-hog-study)
   - [Module 4: Flip Test-Time Augmentation (`detect_tta.m`)](#module-4-flip-test-time-augmentation-detect_ttam)
   - [Module 5: Part-Consistent Score-Weighted Pose Voting (`pose_vote.m`)](#module-5-part-consistent-score-weighted-pose-voting-pose_votem)
   - [Module 6: Anthropometric Kinematic Refinement (`refine_pose.m`)](#module-6-anthropometric-kinematic-refinement-refine_posem)
   - [Module 7: Dataset Transformation & Parsing Back (`PARSE_transback.m`)](#module-7-dataset-transformation--parsing-back-parse_transbackm)
5. [Evaluation Metrics: PCK & APK Explained Mathematically](#5-evaluation-metrics-pck--apk-explained-mathematically)
6. [Complete Experimental Results & Exact Numbers](#6-complete-experimental-results--exact-numbers)
7. [Ablation Study & Negative Findings (Why Certain Ideas Failed)](#7-ablation-study--negative-findings-why-certain-ideas-failed)
8. [The Scientific Protocol & Version History](#8-the-scientific-protocol--version-history)
9. [The Examiner's Viva Hitlist: 25 Tough Questions & Answers](#9-the-examiners-viva-hitlist-25-tough-questions--answers)
10. [Quick-Fire Formula Cheat Sheet](#10-quick-fire-formula-cheat-sheet)

---

# 1. THE 60-SECOND ELEVATOR PITCH & CORE MOTIVATION

### The Problem
Classical pictorial-structure models (specifically Yang & Ramanan's Flexible Mixtures-of-Parts, CVPR 2011) remain celebrated benchmarks in computer vision because they are globally optimal, interpretable, deterministic, and lightweight on CPU without requiring GPUs. However, because they depend on hand-crafted Histogram of Oriented Gradients (HOG) features, their performance collapses catastrophically when images suffer from **sensor noise (-6.6 pp PCK)**, **defocus/motion blur (-6.5 pp PCK)**, or **underexposure**.

### Deep Learning vs. Our Approach
While deep convolutional networks (e.g., DeepPose, HRNet) improve robustness, they require:
1. Massive labelled datasets (millions of parameters),
2. Power-hungry GPU hardware,
3. Complete black-box retraining.

### Our Solution
We asked: **Can we make classical pose estimation robust under hostile imaging conditions entirely at test-time, without retraining the model and without GPUs?**

We wrapped the original, unchanged FMP model in a four-stage test-time enhancement pipeline:
1. **No-Reference Image Quality Assessment:** Fast mathematical extraction of noise ($\sigma$), contrast-normalised sharpness, brightness, and robust contrast.
2. **Quality-Gated Preprocessing:** One targeted restoration per image (Non-Local Means for noise, Richardson-Lucy deconvolution for blur, gamma correction for low brightness).
3. **Flip Test-Time Augmentation (TTA):** Forward and horizontally mirrored passes with bilateral kinematic remapping.
4. **Part-Consistent Pose Voting:** Softmax score-weighted spatial averaging over candidate detections that agree on individual joint hypotheses, eliminating HOG quantization jitter.
5. **Anthropometric Kinematic Refinement:** Anatomical bone length bounds (Winter 2009 / Drillis & Contini 1966) and bilateral symmetry priors.

### The Impact
On held-out test images (PARSE 101–205), our pipeline achieves:
* **+5.99 pp PCK (+8.12 pp APK)** on Gaussian noise (recovering ~90% of lost accuracy).
* **+6.80 pp PCK (+10.71 pp APK)** on blur (fully restoring clean performance).
* **+2.86 pp PCK (+2.17 pp APK)** on dark images.
* **+1.70 pp PCK (+2.12 pp APK)** cross-dataset generalization on BUFFY without any BUFFY-specific tuning.
* **Zero regression on clean images (+1.16 pp PCK)**.
* Inference runs at **~2.4 seconds per image on standard CPU**.

---

# 2. BASE PAPER THEORY: YANG & RAMANAN FMP MODEL (DEEP DIVE & SIMPLE WORDS)

To defend this project, you must know the base paper inside out. The examiner will test if you understand the foundation.

### Paper Citation
* **Authors:** Yi Yang and Deva Ramanan (UC Irvine).
* **Title:** *"Articulated Pose Estimation using Flexible Mixtures of Parts"*, CVPR 2011 / IEEE TPAMI 2013.

---

### 2.0 WHAT IS HOG? (THE INTUITIVE FOUNDATION IN SIMPLE WORDS)

Before touching any mathematical equations, you must understand what **HOG (Histogram of Oriented Gradients)** is, because the entire base paper relies on it.

#### 1. Why Not Just Use Raw RGB Pixel Colors?
Imagine trying to write an algorithm to detect an elbow in a photo:
* Person A wears a bright red shirt.
* Person B wears a dark blue shirt.
* Person C is standing in direct sunlight; Person D is in deep shadow.

If your code checks raw RGB numbers (pixel colors), it gets completely confused because clothing colors and sunlight change constantly.

#### 2. The Big Idea: Look at "Edges", Not Colors!
Whether a shirt is red, black, or blue, **the physical boundary (the edge) of the arm against the background remains identical!**
* An **edge** is simply a sudden change in pixel brightness.
* In mathematics, a sudden change is called a **gradient** ($\nabla I$).
* A vertical limb has vertical edges.
* A tilted forearm has diagonal edges.

#### 3. Step-by-Step: How a HOG Descriptor is Built

```
  Raw Pixels (4x4 box)      Gradient Vectors (Angles)      Histogram (Votes in Bins)
  ┌────────┬────────┐             ↖   ↑   ↗               |
  │ Bright │  Dark  │   ──►       ←   •   →        ──►    |     █
  │ Bright │  Dark  │             ↙   ↓   ↘               |  █  █  █  
  └────────┴────────┘                                     └──┴──┴──┴──
   Sudden edge detected        Direction & Strength        Count in 9 directions
```

1. **Calculate Gradients for Every Pixel:**
   * At every pixel $(x, y)$, calculate:
     * **Strength (Magnitude):** How sharp is the brightness jump? $r = \sqrt{dx^2 + dy^2}$
     * **Direction (Orientation):** Which way is the boundary pointing? $\theta = \arctan(dy / dx)$
2. **Divide the Image into Tiny "Cells" ($4 \times 4$ pixels):**
   * Instead of keeping every pixel, group pixels into tiny $4 \times 4$ pixel squares called **cells**.
3. **Build the Histogram (The "Voting" Step):**
   * Inside each $4 \times 4$ cell (16 pixels), let the pixels "vote" into direction bins (e.g., $0^\circ, 20^\circ, 40^\circ, \dots, 160^\circ$).
   * A strong vertical edge casts a heavy vote into the $90^\circ$ bin.
4. **Block Normalization (Crucial for Viva!):**
   * Group $2 \times 2$ neighboring cells into a **block**.
   * Divide the vector by the total energy of the block:
     $$\mathbf{f} = \frac{\mathbf{v}}{\sqrt{\|\mathbf{v}\|^2 + \epsilon}}$$
   * **Why this is magic:** If sunlight drops by 50%, all pixel gradients drop by 50%. But dividing by the total block energy **cancels out the lighting factor completely!** That is why HOG is inherently invariant to global illumination and contrast changes.
5. **The 32 Channels of Felzenszwalb HOG:**
   * Each cell ends up described by a clean vector of **32 numbers**:
     * 18 contrast-sensitive directed angles ($0^\circ - 360^\circ$),
     * 9 contrast-insensitive undirected angles ($0^\circ - 180^\circ$),
     * 4 block-normalised gradient texture energies,
     * 1 boundary truncation flag (indicating if the cell touches the image boundary).

---

### 2.1 THE CORE IDEA: WHAT IS "FLEXIBLE MIXTURES-OF-PARTS"?

Older classical models (like Felzenszwalb & Huttenlocher 2005) modeled a person as 10–14 rigid rectangular cardboard cutouts (full torso, whole arm, whole leg). These failed whenever joints bent at sharp angles or foreshortened toward the camera.

Yang & Ramanan revolutionized this by introducing three concepts:

#### 1. Mini-Parts (Part Granularity: 26 Parts Instead of 14)
* The evaluation benchmark (PARSE) only tests 14 joints. But the FMP model creates **26 smaller parts**:
  * Joints (shoulders, elbows, wrists, hips, knees, ankles, neck, head).
  * **Midpoints of limbs** (mid-upper arm, mid-forearm, mid-thigh, mid-calf).
* **Why midpoints?** It breaks a stiff rigid limb into two smaller, flexible segments that can bend, flex, and rotate independently without needing complex loops.

#### 2. Mixture of Types ($m_i$)
* A body part does not look the same in all poses.
* For each part $i$, the model allows a discrete mixture type $m_i \in \{1, 2, \dots, K\}$ (typically $K=6$).
* Think of $m_i$ as **different appearance stencils**: e.g., an outstretched horizontal arm stencil, a vertically hanging arm stencil, a $45^\circ$ bent arm stencil.

#### 3. Co-occurrence & Kinematic Springs
* The parts are connected in a **tree-structured graph** (no closed loops).
* Virtual springs connect parent parts to child parts, penalizing unphysical stretching.
* Co-occurrence tables penalize anatomically impossible combinations of angles.

---

### 2.2 THE MASTER MATHEMATICAL ENERGY FORMULATION (EVERY SYMBOL EXPLAINED)

A complete human pose is represented by two sets of variables:
1. **$L = \{l_1, l_2, \dots, l_{26}\}$**: The pixel locations of all 26 parts, where each $l_i = (x_i, y_i)$.
2. **$M = \{m_1, m_2, \dots, m_{26}\}$**: The mixture orientation type assigned to each part, where $m_i \in \{1, \dots, K\}$.

The total score of placing pose $(L, M)$ onto image $I$ is given by:

$$S(I, L, M) = \sum_{i \in V} q_i(I, l_i, m_i) + \sum_{(i,j) \in E} \psi_{ij}(l_i, l_j, m_i, m_j) + S(M)$$

#### Symbol Translation Dictionary:
| Symbol | Full Name | Plain English Translation |
|:---|:---|:---|
| $S(I, L, M)$ | **Total Energy Score** | How confident the model is that this is a real human in this exact pose. **Higher score = better detection.** |
| $I$ | **Input Image** | The raw photograph pixels. |
| $L$ | **Locations** | The pixel coordinates $(x, y)$ of all 26 body parts. |
| $M$ | **Mixtures** | The orientation stencils selected for all 26 body parts. |
| $V$ | **Vertices (Parts)** | The set of all 26 body parts ($V = \{1, 2, \dots, 26\}$: Head, Neck, Elbows, Knees, etc.). |
| $E$ | **Edges (Connections)** | The kinematic tree links (bones) connecting parent and child parts (e.g., Shoulder $\rightarrow$ Elbow). |

---

### DEEP-DIVE INTO THE THREE TERMS:

```
 Total Score S  =  [ Term 1: Appearance ]  +  [ Term 2: Spring Deformation ]  +  [ Term 3: Compatibility Prior ]
                   "Does it look like an       "Are the limbs physically         "Do these joint angles make
                    elbow at this pixel?"       connected at proper lengths?"     sense together anatomically?"
```

#### TERM 1: The Unary Appearance Term — $\sum_{i \in V} q_i(I, l_i, m_i)$
> **Simple Intuition:** *"Pattern Matching / Stencil Test."*

$$q_i(I, l_i, m_i) = w_i^{m_i} \cdot \phi(I, l_i)$$

* **$\phi(I, l_i)$**: The 32-dimensional HOG feature vector extracted from image $I$ at pixel location $l_i = (x_i, y_i)$.
* **$w_i^{m_i}$**: The trained linear template filter (support vector machine weights) for part $i$ with mixture type $m_i$.
* **The Dot Product ($\cdot$):**
  * It measures pattern alignment:
    * If the image edges match the template stencil edges $\rightarrow$ **Large positive score!**
    * If you place an elbow stencil over flat grass $\rightarrow$ **Near-zero or negative score.**
* **The Sum $\sum_{i \in V}$:** We add up the appearance match scores for all 26 parts.

---

#### TERM 2: The Pairwise Deformation (Spring) Term — $\sum_{(i,j) \in E} \psi_{ij}(l_i, l_j, m_i, m_j)$
> **Simple Intuition:** *"Hooke's Law Physics Springs."*

$$\psi_{ij}(l_i, l_j, m_i, m_j) = w_{ij}^{m_i, m_j} \cdot [dx, dy, dx^2, dy^2]^T$$

Where:
* $dx = x_i - x_j$ (horizontal pixel distance between child part $i$ and parent part $j$).
* $dy = y_i - y_j$ (vertical pixel distance between child part $i$ and parent part $j$).

* **The Physical Spring Analogy:**
  * Imagine an invisible physical spring connecting the **Shoulder** to the **Elbow**.
  * The spring has a natural resting position (the anchor offset).
  * If the elbow is located near its expected anatomical distance $\rightarrow$ Displacement is minimal, penalty is zero.
  * If the elbow drifts too far away or detaches $\rightarrow$ The spring stretches!
  * The **$dx^2$ and $dy^2$** terms penalize large displacements quadratically (identical to Hooke's Law in physics: $\text{Energy} = \frac{1}{2} k x^2$).
* **$w_{ij}^{m_i, m_j}$ (Spring Stiffness Vector):**
  * The trained spring parameters: $[w_x, w_y, w_{xx}, w_{yy}]$.
  * For Head-to-Neck: $w$ is **very stiff** (the head cannot detach from the neck).
  * For Wrist-to-Elbow: $w$ is **more flexible** (hands can bend and rotate freely).

---

#### TERM 3: The Mixture Co-occurrence Prior — $S(M)$
> **Simple Intuition:** *"Anatomical Common Sense & Compatibility."*

$$S(M) = \sum_{i \in V} b_i^{m_i} + \sum_{(i,j) \in E} b_{ij}^{m_i, m_j}$$

* **$b_i^{m_i}$ (Single-Part Prior Bias):**
  * How often does this part orientation occur in natural human life?
  * Example: People stand upright with heads pointed up far more often than upside down. The upright head mixture gets a positive bias score; upside down gets a penalty.
* **$b_{ij}^{m_i, m_j}$ (Pairwise Mixture Compatibility):**
  * Do the mixture types of part $i$ and part $j$ physically make sense together?
  * Example: If the upper arm is pointing straight down, can the forearm physically point backward into the torso? No! The human elbow joint cannot bend that way.
  * If two mixture types are physically impossible together, $b_{ij}$ applies a massive negative penalty, preventing grotesque alien poses.

---

### 2.3 HOW INFERENCE RUNS IN ~1 SECOND: DYNAMIC PROGRAMMING & DISTANCE TRANSFORMS

If you have 26 parts, each with thousands of $(x, y)$ pixel coordinates and 6 mixture types, brute-force checking every combination would take $(W \times H \times 6)^{26}$ operations—which would take **billions of years** to compute!

#### How FMP Solves It in ~1 Second on CPU:
Because the body model is a **tree** (branches out from the head, with **no closed kinematic loops**), the global optimum can be found exactly using **Dynamic Programming**:

```
 [Fingers / Toes (Leaves)]   ──►   Pass best scores UP   ──►   [Head (Root)]
                                                                    │
 [Optimal Full Pose]        ◄──   Trace back pointers   ◄──   Find Peak
```

1. **Bottom-Up Message Passing (Leaves to Root):**
   * Start at the leaf parts (wrists, ankles).
   * For every possible location of the parent (elbow), calculate the best location for the child (wrist) using:
     $$\text{msg}(l_{par}) = \max_{l_{\text{child}}} \left[ q(l_{\text{child}}) + \psi(l_{\text{child}}, l_{par}) \right]$$
   * Pass messages upward: Wrist $\rightarrow$ Elbow $\rightarrow$ Shoulder $\rightarrow$ Neck $\rightarrow$ Head.
2. **Generalized Distance Transform (Felzenszwalb & Huttenlocher):**
   * Normally, searching all $(x, y)$ positions of the child takes $O(W^2 H^2)$ (quadratic in pixels).
   * Because the spring penalties are independent parabolas ($dx^2, dy^2$), the 2D search decouples into two 1D lower-envelope calculations.
   * This computes the exact global maximum in **$O(WH)$ linear time**!
3. **Top-Down Traceback (Root to Leaves):**
   * At the head (root), find the pixel with the highest accumulated score.
   * Follow the recorded back-pointers down the tree to immediately extract the exact $(x, y)$ coordinates of all 26 parts.
4. **Parsing Back to 14 Benchmark Joints (`PARSE_transback.m`):**
   * After finding the 26 parts, a simple linear projection matrix drops the 12 intermediate midpoints, outputting the clean 14 standard benchmark joints for evaluation.

---

### 2.4 EXACT VIVA PHRASES TO SPEAK OUT LOUD

* **If asked: "What is $q_i$ in your formula?"**
  > *"That is the unary appearance term. It is the dot product of the 32-dimensional Felzenszwalb HOG feature vector at that pixel with the trained linear SVM template for that part and mixture type. It measures how much that image patch resembles that body part."*

* **If asked: "What is $\psi_{ij}$?"**
  > *"That is the pairwise deformation term. It models the kinematic connection between parent and child parts as a quadratic spring: $w \cdot [dx, dy, dx^2, dy^2]^T$. It penalizes displacement from the natural anatomical resting position, acting like Hooke's Law."*

* **If asked: "What is $S(M)$?"**
  > *"That is the mixture co-occurrence prior. $b_i$ is the prior probability of seeing that part orientation, and $b_{ij}$ enforces compatibility between connected parts—penalizing anatomically impossible joint angle combinations."*

* **If asked: "Why 26 parts when PARSE only evaluates 14 joints?"**
  > *"Yang & Ramanan introduced 12 intermediate midpoints along the limbs. This breaks rigid bones into smaller flexible segments that capture foreshortening and clothing flexion without needing loops in the graph. After inference, `PARSE_transback.m` extracts the 14 true joints."*

---

# 3. END-TO-END SYSTEM ARCHITECTURE

```
                      ┌──────────────────────────────────────────────┐
                      │             Raw Input Image (RGB)            │
                      └──────────────────────┬───────────────────────┘
                                             │
                                             ▼
                      ┌──────────────────────────────────────────────┐
                      │ [Stage 1] Image Quality Analysis             │
                      │  - Brightness μ = mean(g)/255                │
                      │  - Robust Contrast = (p99 - p1)/255          │
                      │  - Noise σ: Immerkaer (1996) 3x3 Laplacian   │
                      │  - Sharpness: Normalized Laplacian Energy    │
                      └──────────────────────┬───────────────────────┘
                                             │
                                             ▼
                      ┌──────────────────────────────────────────────┐
                      │ [Stage 2] Priority-Gated Restoration         │
                      │  Decision Rule:                              │
                      │  ├─ If σ > 6.0       ──► Non-Local Means     │
                      │  ├─ Else If S < 0.014──► Richardson-Lucy     │
                      │  ├─ Else If μ < 0.40 ──► Gamma (γ = 0.67)    │
                      │  └─ Otherwise        ──► Pass Through        │
                      └──────────────────────┬───────────────────────┘
                                             │
                                             ▼
                      ┌──────────────────────────────────────────────┐
                      │ [Stage 3] HOG Feature Pyramid (Unchanged)    │
                      │  - 32-D Felzenszwalb HOG cells               │
                      │  - 10 levels per octave                      │
                      └──────────────────────┬───────────────────────┘
                                             │
                                             ▼
                      ┌──────────────────────────────────────────────┐
                      │ [Stage 4] FMP Inference + Flip TTA           │
                      │  - Pass 1: Original restored image           │
                      │  - Pass 2: Horizontally mirrored image       │
                      │  - Re-project coordinates: x' = W + 1 - x    │
                      │  - Swap bilateral part indices               │
                      │  - Pool top-500 candidates                   │
                      └──────────────────────┬───────────────────────┘
                                             │
                                             ▼
                      ┌──────────────────────────────────────────────┐
                      │ [Stage 4b] Part-Consistent Pose Voting       │
                      │  - Select best detection box* (score s*)     │
                      │  - Filter cluster: IoU > 0.5 & s >= s* - 1.0 │
                      │  - Softmax weight: w_i = exp((s_i - s*)/0.5) │
                      │  - Part gate: ||P_j - P_j*|| <= 0.25 * Size  │
                      │  - Compute weighted mean for each joint      │
                      └──────────────────────┬───────────────────────┘
                                             │
                                             ▼
                      ┌──────────────────────────────────────────────┐
                      │ [Stage 5] Dataset Mapping & Refinement       │
                      │  - PARSE_transback: 26 parts -> 14 joints    │
                      │  - Reference Torso Scale + 30% Height Prior  │
                      │  - Anthropometric Bone Bounds (Drillis 1966) │
                      │  - Bilateral Symmetry Projection             │
                      └──────────────────────┬───────────────────────┘
                                             │
                                             ▼
                      ┌──────────────────────────────────────────────┐
                      │             Final Refined Pose               │
                      └──────────────────────────────────────────────┘
```

---

# 4. LINE-BY-LINE CODE & MODULE WALKTHROUGH

## MODULE 1: IMAGE QUALITY ASSESSMENT (`image_quality.m`)
**File:** `code_full/image_quality.m`  
**Purpose:** Measure image characteristics without reference ground-truth to trigger the proper restorative operator.

```matlab
function q = image_quality(im)
    if ~isa(im, 'uint8'), im = im2uint8(im); end
    if size(im, 3) == 3, g = rgb2gray(im); else, g = im; end
    g = double(g);
    [H, W] = size(g);

    % 1. Brightness: Normalized mean luminance
    q.brightness = mean(g(:)) / 255;

    % 2. Contrast: Robust percentile range (ignoring specular outliers)
    p = prctile(g(:), [1 99]);
    q.contrast = (p(2) - p(1)) / 255;

    % 3. Noise Estimation: Immerkaer (1996) Fast Laplacian Estimator
    M = [ 1 -2  1; 
         -2  4 -2; 
          1 -2  1];
    q.noiseSigma = sum(sum(abs(conv2(g, M, 'valid')))) * sqrt(pi / 2) / (6 * (W - 2) * (H - 2));

    % 4. Sharpness: Contrast-Normalized Laplacian Energy
    % First apply light Gaussian smoothing (sigma=0.8) to remove sensor noise
    gs  = imgaussfilt(g, 0.8);
    lap = conv2(gs, [0 1 0; 1 -4 1; 0 1 0], 'valid');
    q.sharpness = var(lap(:)) / max(var(g(:)), 1);
end
```

### Deep-Dive Explanations:
1. **Why Immerkær's Estimator?**
   * *Formula:* $M = \begin{bmatrix} 1 & -2 & 1 \\ -2 & 4 & -2 \\ 1 & -2 & 1 \end{bmatrix}$.
   * *Mathematical Property:* This kernel cancels all polynomial image structure up to second order (constant, linear slopes, and quadratic curves). Thus, it responds **purely to high-frequency zero-mean Gaussian noise** and is immune to image edges and textures!
   * *Scaling factor:* $\frac{\sqrt{\pi/2}}{6}$ comes from the expectation of the folded normal distribution $E[|X|] = \sigma \sqrt{2/\pi}$, where the sum of squares of kernel elements is $\sum M_{ij}^2 = 36$, so $\sqrt{36} = 6$.
   * *Result:* On PARSE test images, clean images score $\sigma \le 9.2$, while noisy images score $\sigma \ge 9.9$. Perfect separation!
2. **Why Contrast-Normalized Laplacian for Sharpness?**
   * Standard blur detectors measure variance of the Laplacian: $\text{Var}(\nabla^2 I)$.
   * *The Fatal Flaw:* A low-contrast image naturally has small gradients, so $\text{Var}(\nabla^2 I)$ is tiny, causing low-contrast images to be falsely diagnosed as blurry!
   * *Our Fix:* Dividing by the image variance $\text{Var}(I)$ makes sharpness **scale- and contrast-invariant**.
   * Pre-filtering with `imgaussfilt(g, 0.8)` ensures high-frequency noise spikes don't masquerade as edge sharpness.

---

## MODULE 2: GATED ADAPTIVE RESTORATION (`adaptive_preprocess.m`)
**File:** `code_full/adaptive_preprocess.m`  
**Purpose:** Apply exactly one targeted restorative filter based on the image quality metrics.

```matlab
function [out, info] = adaptive_preprocess(im, params)
    if nargin < 2, params = struct(); end
    params = set_default(params, 'noiseSigmaThresh', 6.0);
    params = set_default(params, 'sharpnessThresh', 0.014);
    params = set_default(params, 'brightnessThresh', 0.40);
    params = set_default(params, 'gamma', 0.67);
    params = set_default(params, 'deblurSigma', 1.5);
    params = set_default(params, 'deblurIters', 8);

    originalClass = class(im);
    if ~isa(im, 'uint8'), im = im2uint8(im); end

    q = image_quality(im);
    info.brightness = q.brightness;
    info.contrast   = q.contrast;
    info.noise      = q.noiseSigma;
    info.sharpness  = q.sharpness;
    info.operations = {};

    % Priority-Gated Decision Ladder
    if q.noiseSigma > params.noiseSigmaThresh
        enhanced = imnlmfilt(im);
        info.operations{end+1} = 'Non-local-means denoising';
    elseif q.sharpness < params.sharpnessThresh
        psf = fspecial('gaussian', 9, params.deblurSigma);
        enhanced = im2uint8(deconvlucy(im2double(im), psf, params.deblurIters));
        info.operations{end+1} = 'Richardson-Lucy deblurring';
    elseif q.brightness < params.brightnessThresh
        enhanced = imadjust(im, [], [], params.gamma);
        info.operations{end+1} = 'Gamma correction';
    else
        enhanced = im;
    end
    info.numOperations = numel(info.operations);
...
```

### Deep-Dive Explanations:
1. **Why the Priority Order (Noise > Blur > Dark)?**
   * High-frequency noise is destructive to HOG gradients. If an image is both noisy and blurry, running deconvolution first will massively amplify the noise! Therefore, **noise must take highest priority**.
2. **Why Non-Local Means (NLM) instead of Bilateral / Gaussian Filter?**
   * A Bilateral filter averages pixels within a local spatial window based on photometric similarity. At moderate noise ($\sigma \approx 17$), a bilateral filter either leaves grain or, with larger radiuses, smears thin limbs (e.g. wrists and ankles).
   * **Non-Local Means (`imnlmfilt`)** searches across the entire search window for *similar patch geometries*. It preserves repetitive texture and sharp limb boundaries while destroying zero-mean noise.
   * Tuning PCK: NLM achieved **70.7%**, beating Bilateral (69.1%), Gaussian (69.0%), and Wiener (68.8%).
3. **Why Richardson-Lucy Deconvolution instead of Unsharp Masking?**
   * Unsharp masking simply amplifies high-frequency differences: $I_{\text{sharp}} = I + \alpha(I - G_\sigma * I)$. It cannot hallucinate or recover spatial frequencies destroyed by blur; it merely creates ringing halos around degraded edges.
   * **Richardson-Lucy** is an iterative, non-linear Bayesian maximum likelihood deconvolution algorithm under Poisson noise:
     $$I^{(t+1)} = I^{(t)} \cdot \left( \frac{I_{\text{blurred}}}{I^{(t)} * \text{PSF}} * \text{PSF}^T \right)$$
   * We calibrate with a Gaussian Point Spread Function ($\text{PSF}$, $\sigma=1.5$, size 9x9) for 8 iterations.
   * Tuning PCK: Richardson-Lucy achieved **68.3%**, whereas unsharp masking scored **65.9%** (worse than baseline 66.1%).
4. **Why Gamma Correction ($\gamma = 0.67$)?**
   * For dark images ($\mu < 0.40$), linear scaling clips highlights. Gamma mapping $s = r^\gamma$ with $\gamma < 1.0$ compresses highlights while expanding shadow details, pulling gradients out of dark background regions.

---

## MODULE 3: FEATURE EXTRACTION & THE ADAPTIVE HOG STUDY
**File:** `code_basic/adaptive_features.cc` & `code_basic/adaptive_detect_fast.m`  
**Purpose:** Investigation of adaptive HOG feature calculation.

### How HOG Works in FMP:
1. Image is converted into a feature pyramid: scales $s = 2^{i/\text{interval}}$, with 10 levels per octave.
2. Cells of size $sbin = 4 \times 4$ pixels.
3. For each pixel, compute gradient orientation $\theta = \arctan(dy/dx)$ and magnitude $r = \sqrt{dx^2 + dy^2}$.
4. Bilinear interpolation of gradients into 9 contrast-insensitive orientations ($0^\circ - 180^\circ$) and 18 contrast-sensitive orientations ($0^\circ - 360^\circ$).
5. Energy normalisation over 4 different $2 \times 2$ cell blocks:
   $$v_{\text{norm}} = \frac{v}{\sqrt{\|v\|^2 + \epsilon}}$$
6. Yields **32 channels**: 18 directed orientations + 9 undirected orientations + 4 block texture features + 1 boundary truncation flag.

### The Viva Crucial Finding: Why Adaptive HOG FAILED Without Retraining
* In `adaptive_features.cc`, gradient weighting and local adaptation were attempted.
* **The Result:** Accuracy remained identical to baseline ($\pm 0.1$ pp PCK) across all conditions.
* **The Mathematical Reason:** The FMP part templates $w_i^{m_i}$ are linear Support Vector Machine (SVM) weights trained on the *exact distribution* of standard Felzenszwalb HOG.
  $$S_i(x) = w_i^T \cdot x + b_i$$
  When feature values $x$ are modified at test time without retraining $w_i$, you introduce a **distribution shift**. The linear dot product no longer aligns with the learned hyperplanes!
* Furthermore, changing cell size (e.g. from 4 to 8) or orientation bins changes the *dimensionality* of $x$, making dot products with pre-trained 32-D templates mathematically impossible without full model retraining.
* Full retraining requires the INRIA Person negative dataset (1,218 background images), which was beyond the test-time scope.

---

## MODULE 4: FLIP TEST-TIME AUGMENTATION (`detect_tta.m`)
**File:** `code_full/detect_tta.m`  
**Purpose:** Exploit bilateral reflection invariance to generate a richer candidate pool.

```matlab
function boxes = detect_tta(im, model, thresh, topK)
    if nargin < 4, topK = 500; end

    % Pass 1: Standard image detection
    boxes  = keep_top(detect_fast(im, model, thresh), topK);

    % Pass 2: Horizontally flipped image detection
    boxesF = keep_top(detect_fast(im(:, end:-1:1, :), model, thresh), topK);
    if isempty(boxesF), return; end

    numparts = floor((size(boxesF, 2) - 2) / 5);
    half = (numparts - 2) / 2;
    % Mirror index mapping for PARSE: parts 1-2 are head/neck; 3:14 and 15:26 swap!
    mirror = [1 2, (3 + half):numparts, 3:(2 + half)];

    W = size(im, 2);
    % Reflect X-coordinates back to original image space
    for p = 1:numparts
        c = 5 * (p - 1);
        x1 = W + 1 - boxesF(:, c + 3);
        x2 = W + 1 - boxesF(:, c + 1);
        boxesF(:, c + 1) = x1;
        boxesF(:, c + 3) = x2;
    end
    % Swap left and right body parts
    cols = reshape(1:5 * numparts, 5, numparts);
    cols = cols(:, mirror);
    boxesF(:, 1:5 * numparts) = boxesF(:, cols(:)');

    % Pool both candidate sets together
    boxes = [boxes; boxesF];
end
```

### Deep-Dive Explanations:
1. **Coordinate Reflection Math:**
   For an image of width $W$, a point at $x_f$ in the flipped image corresponds to:
   $$x_{\text{orig}} = W + 1 - x_f$$
   Since bounding boxes have $x_1 \le x_2$, reflecting reverses the order:
   $$x_1^{\text{orig}} = W + 1 - x_2^f, \quad x_2^{\text{orig}} = W + 1 - x_1^f$$
2. **Kinematic Swapping:**
   In the 26-part PARSE model:
   * Parts 1–2: Head top and Neck (central line, no swap).
   * Parts 3–14: Right body parts (shoulder, arm, hand, hip, leg, foot).
   * Parts 15–26: Left body parts.
   * `mirror = [1 2 15:26 3:14]`. When the person is mirrored, their physical left arm appears on the right side of the image, so part identities must be explicitly permuted!

---

## MODULE 5: PART-CONSISTENT SCORE-WEIGHTED POSE VOTING (`pose_vote.m`)
**File:** `code_full/pose_vote.m`  
**Purpose:** Eliminate discrete HOG grid quantization jitter and mixture-switching noise by averaging spatial configurations across high-confidence overlapping candidates.

```matlab
function obj = pose_vote(box, boxes, params)
    if nargin < 3, params = struct(); end
    if ~isfield(params, 'iou'),   params.iou = 0.5;   end
    if ~isfield(params, 'delta'), params.delta = 1.0; end
    if ~isfield(params, 'T'),     params.T = 0.5;     end
    if ~isfield(params, 'r'),     params.r = 0.25;    end

    obj = box2obj(box);
    s = boxes(:, end);
    % Find near-duplicate candidates of the same person
    keep = person_iou(box, boxes) > params.iou & s >= box(end) - params.delta;
    if nnz(keep) < 2, return; end

    % Softmax score weights
    w = exp((s(keep) - box(end)) / params.T);
    cand = box2obj(boxes(keep, :));
    P = cat(3, cand.point);                                   % parts x 2 x K
    W = repmat(reshape(w, 1, 1, []), size(P, 1), 1, 1);       % parts x 1 x K

    % Part-Consistency Gate: Only allow voting if candidate agrees on that specific part
    if isfinite(params.r)
        sz = max(max(obj.point) - min(obj.point));             % Overall person scale
        d = sqrt(sum((P - obj.point) .^ 2, 2));                % Distance per part
        W = W .* (d <= params.r * sz);                         % Gated weights
    end

    % Normalized weighted spatial mean
    obj.point = sum(P .* W, 3) ./ sum(W, 3);
end
```

### Deep-Dive Explanations:
1. **The Problem with Standard Arg-Max Detection:**
   FMP outputs a single discrete peak location on the HOG cell grid. Often, 5–10 other candidate detections exist with almost identical scores ($s \approx s^*$), shifted by $\pm 1$ cell or using a different mixture component.
2. **The Failure of Naive "Whole-Pose" Voting (Version 2 Bug):**
   * If candidate $A$ detects an arm raised, and candidate $B$ detects the arm lowered, naive averaging blends them into an unphysical pose in mid-air!
   * Worse: On blurry images, naive voting pulled the head off-target (**head PCK dropped by -4.5 pp**).
3. **The Innovation: Part-Consistent Voting:**
   * We enforce: A candidate votes for joint $j$ **if and only if** its predicted joint $j$ lies within $r = 0.25 \times \text{person\_scale}$ of the selected detection's joint $j$:
     $$\|P_j^{(i)} - P_j^*\| \le 0.25 \times S$$
   * If a candidate represents a completely different limb hypothesis, its weight for that joint is zeroed out!
   * *Impact:* Raised the tuning split mean PCK from 73.2% to **75.0%**.

---

## MODULE 6: ANTHROPOMETRIC KINEMATIC REFINEMENT (`refine_pose.m`)
**File:** `code_full/refine_pose.m`  
**Purpose:** Enforce biological and physical constraints on joint coordinates based on human biomechanics literature.

```matlab
function [refined_points, info] = refine_pose(points, params)
...
    % Anthropometric bounds calibrated from Drillis & Contini (1966) & Winter (2009)
    params.maxUpperArmRatio = 0.90; params.minUpperArmRatio = 0.35;
    params.maxForearmRatio  = 0.80; params.minForearmRatio  = 0.30;
    params.maxThighRatio    = 1.20; params.minThighRatio    = 0.50;
    params.maxCalfRatio     = 1.15; params.minCalfRatio     = 0.45;
    params.symmetryWeight   = 0.35;

    % 1. Robust Torso Scale Calculation
    midShoulder  = (points(9, :) + points(10, :)) / 2;
    midHip       = (points(3, :) + points(4, :)) / 2;
    torsoLen     = norm(midShoulder - midHip);

    % Anchor with overall height prior (~30% of standing height)
    personHeight = max(max(points(:,2)) - min(points(:,2)), 50.0);
    priorTorsoScale = 0.30 * personHeight;

    if torsoLen >= 0.18 * personHeight && torsoLen <= 0.45 * personHeight
        torsoScale = 0.65 * torsoLen + 0.35 * priorTorsoScale;
    else
        torsoScale = priorTorsoScale;
    end

    % 2. Target Limb Lengths with Bilateral Symmetry
    targetUpper = median([lenUpperR, lenUpperL, 0.60 * torsoScale]);
    targetFore  = median([lenForeR,  lenForeL,  0.50 * torsoScale]);
    targetThigh = median([lenThighR, lenThighL, 0.85 * torsoScale]);
    targetCalf  = median([lenCalfR,  lenCalfL,  0.75 * torsoScale]);

    % 3. Sequential Projection along Kinematic Chains
    % Shoulder -> Elbow -> Wrist
    % Hip -> Knee -> Ankle
...
```

### Deep-Dive Explanations:
1. **Drillis & Contini (1966) & David Winter (2009) Biomechanics:**
   In human body segment parameter literature, segment lengths are proportional to body height:
   * Upper arm $\approx 0.186 \times H$
   * Forearm $\approx 0.146 \times H$
   * Thigh $\approx 0.245 \times H$
   * Calf $\approx 0.246 \times H$
   * Torso $\approx 0.300 \times H$
   Expressed as ratios of torso length: Upper arm $\approx 0.60$, Forearm $\approx 0.50$, Thigh $\approx 0.85$, Calf $\approx 0.75$.
2. **Robust Torso Scale Anchoring:**
   * If a shoulder or hip detection slips, measuring raw torso distance $\text{dist}(\text{shoulder}, \text{hip})$ produces a broken scale that distorts every other limb in the body.
   * We anchor torso scale using a robust prior: $0.30 \times (\text{detected vertical height})$. If measured torso length is outside $[0.18H, 0.45H]$, it falls back strictly to the prior!
3. **Bilateral Symmetry Prior:**
   Left and right limbs on a human body have identical physical lengths:
   $$\text{target\_length} = \text{median}(L_{\text{left}}, L_{\text{right}}, \text{ratio} \times \text{torsoScale})$$
   Taking the median of the three values ensures an outlier on one side is rejected by the agreement between the other side and the anthropometric prior!

---

## MODULE 7: DATASET TRANSFORMATION & PARSING BACK (`PARSE_transback.m`)
**File:** `code_full/PARSE_transback.m`  
**Purpose:** Map 26 FMP model part locations to the 14 standard ground-truth evaluation joints.

```matlab
function det = PARSE_transback(det)
    % Map 26 model parts to 14 standard PARSE benchmark joints
    I = [1  2  3  4  5  6  7  8  9  10 11 12 13 14];
    J = [14 12 10 22 24 26 7  5  3  15 17 19 2  1];
    A = [1  1  1  1  1  1  1  1  1  1  1  1  1  1];
    Transback = full(sparse(I, J, A, 14, 26));

    for n = 1:numel(det)
        for i = 1:numel(det(n).obj)
            if isempty(det(n).obj(i).point), continue, end
            det(n).obj(i).point = Transback * det(n).obj(i).point;
        end
    end
end
```

### The 14 Standard Benchmark Joints:
| ID | Joint Name | Source FMP Part |
|:---|:---|:---|
| 1 | Right Ankle | Part 14 |
| 2 | Right Knee | Part 12 |
| 3 | Right Hip | Part 10 |
| 4 | Left Hip | Part 22 |
| 5 | Left Knee | Part 24 |
| 6 | Left Ankle | Part 26 |
| 7 | Right Wrist | Part 7 |
| 8 | Right Elbow | Part 5 |
| 9 | Right Shoulder | Part 3 |
| 10 | Left Shoulder | Part 15 |
| 11 | Left Elbow | Part 17 |
| 12 | Left Wrist | Part 19 |
| 13 | Neck | Part 2 |
| 14 | Head Top | Part 1 |

---

# 5. EVALUATION METRICS: PCK & APK EXPLAINED MATHEMATICALLY

The examiner will definitely ask you to write the definitions of **PCK** and **APK** on paper/board.

### 5.1 PCK (Percentage of Correct Keypoints)
* **Threshold Criteria:** A predicted joint $\hat{p}_j = (\hat{x}_j, \hat{y}_j)$ is marked as a **True Positive (TP)** if its Euclidean distance to the ground truth keypoint $p_j = (x_j, y_j)$ is less than a fraction $\alpha$ of the person's bounding box scale $S$:
  $$\|\hat{p}_j - p_j\|_2 \le \alpha \cdot S$$
* **Standard Scale Definition:**
  $$S = \max(W_{\text{person}}, H_{\text{person}})$$
* **Benchmark Thresholds:**
  * For PARSE: $\alpha = 0.1$ (strict criterion: within 10% of body size).
  * For BUFFY: $\alpha = 0.2$ (standard upper-body criterion).
* **Overall PCK Calculation:**
  $$\text{PCK}_j = \frac{\sum_{i=1}^N \mathbb{I}(\|\hat{p}_{j,i} - p_{j,i}\|_2 \le \alpha \cdot S_i)}{N}$$
  Mean PCK is the average of $\text{PCK}_j$ across all joints.

### 5.2 APK (Average Precision of Keypoints)
* Introduced by Yang & Ramanan (PAMI 2013) to penalize false detections.
* Unlike PCK (which only measures localization on known detections), **APK measures detection and pose estimation jointly**.
* Candidates are ranked by detection confidence score $s$.
* For a given candidate, a keypoint is a True Positive if it satisfies the distance threshold and has not already been claimed by a higher-scoring candidate (IoU overlap).
* APK is the area under the Precision-Recall curve:
  $$\text{APK} = \int_0^1 P(R) \, dR = \sum_{k=1}^K P(k) \cdot \Delta R(k)$$

### 5.3 Seven-Joint Symmetry Reduction (`to7`)
In official PARSE evaluation, left and right bilateral joints are averaged together to report 7 anatomical parts:
1. **Head** = mean(Head Top, Neck)
2. **Shoulder** = mean(Left Shoulder, Right Shoulder)
3. **Elbow** = mean(Left Elbow, Right Elbow)
4. **Wrist** = mean(Left Wrist, Right Wrist)
5. **Hip** = mean(Left Hip, Right Hip)
6. **Knee** = mean(Left Knee, Right Knee)
7. **Ankle** = mean(Left Ankle, Right Ankle)

---

# 6. COMPLETE EXPERIMENTAL RESULTS & EXACT NUMBERS

You must memorize the headline numbers for the viva.

### 6.1 PARSE Held-Out Images 101–205 (N = 105) — PCK (%)
*This is the uncontaminated test split that was never used for parameter tuning.*

| Condition | Baseline FMP | Adaptive HOG | + Preprocess | + Flip TTA / Vote | **Full Pipeline** | Full − Baseline [95% Bootstrap CI] |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Clean** | 75.37% | 75.44% | 74.83% | 76.53% | **76.53%** | +1.16 pp [−1.53, +3.61] |
| **Dark** | 75.51% | 75.51% | 74.76% | 78.16% | **78.37%** | **+2.86 pp [+0.54, +5.03]** |
| **Low Contrast** | 74.76% | 74.69% | 74.49% | 76.46% | **76.80%** | +2.04 pp [−0.48, +4.22] |
| **Noise (Gaussian)**| 68.78% | 68.64% | 71.02% | 74.83% | **74.76%** | **+5.99 pp [+2.69, +9.12]** |
| **Blur (Defocus)** | 68.91% | 68.91% | 71.16% | 74.97% | **75.71%** | **+6.80 pp [+3.95, +10.07]** |

### 6.2 PARSE Held-Out Images 101–205 — APK (%)

| Condition | Baseline FMP | Full Pipeline | Absolute Gain ($\Delta$) |
|:---|:---:|:---:|:---:|
| **Clean** | 66.19% | **68.07%** | +1.88 pp |
| **Dark** | 67.13% | **69.30%** | +2.17 pp |
| **Low Contrast** | 66.28% | **69.85%** | +3.57 pp |
| **Noise** | 57.26% | **65.38%** | **+8.12 pp** |
| **Blur** | 57.43% | **68.14%** | **+10.71 pp** |

### 6.3 Per-Joint Breakdown on Clean Held-Out PARSE (Baseline $\rightarrow$ Full)
* **Head:** 91.9% $\rightarrow$ 90.5% (−1.4)
* **Shoulder:** 88.6% $\rightarrow$ 87.1% (−1.5)
* **Elbow:** 68.6% $\rightarrow$ **70.5%** (+1.9)
* **Wrist:** 51.4% $\rightarrow$ 47.6% (−3.8)
* **Hip:** 80.0% $\rightarrow$ **83.3%** (+3.3)
* **Knee:** 77.6% $\rightarrow$ **81.4%** (+3.8)
* **Ankle:** 69.5% $\rightarrow$ **75.2%** (+5.7)

*Viva Insight:* Limbs and lower-body joints (ankles, knees, hips, elbows) gain massive precision. The slight wrist and head dips are within the 95% bootstrap confidence intervals ($[-9.5, +1.9]$ for wrist) and represent expected split-to-split variance.

### 6.4 BUFFY Cross-Dataset Benchmark (All 276 Test Frames)
*Evaluated on real TV show frames without ANY tuning on BUFFY!*

| Metric | Head | Shoulder | Elbow | Wrist | Hip | **Average** |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **PCK Baseline** | 96.0% | 97.1% | 87.5% | 67.2% | 85.3% | 86.63% |
| **PCK Enhanced** | 95.7% | 97.1% | 90.6% | 68.8% | 89.5% | **88.33% (+1.70)** |
| **APK Baseline** | 88.3% | 88.8% | 78.6% | 56.2% | 71.2% | 76.63% |
| **APK Enhanced** | 88.3% | 89.1% | 83.1% | 56.5% | 76.9% | **78.75% (+2.12)** |

### 6.5 Runtime and Computational Complexity
Measured on single-core Intel CPU:
* **Quality Analysis + Preprocessing:** $\approx 0.13\text{ s}$ per image (NLM is the main cost).
* **FMP Inference (1 pass):** $\approx 1.25\text{ s}$.
* **FMP Inference with Flip TTA (2 passes):** $\approx 2.35\text{ s}$.
* **Voting + Refinement:** $< 0.01\text{ s}$.
* **Total Runtime:** Baseline $\approx 1.3\text{ s}$ vs. Full Enhanced $\approx 2.4\text{ s}$.

---

# 7. ABLATION STUDY & NEGATIVE FINDINGS (WHY CERTAIN IDEAS FAILED)

Examiners love asking about what didn't work. Demonstrating why certain techniques failed proves you did real research.

### 1. Why CLAHE was tested and rejected:
* *Hypothesis:* Contrast-Limited Adaptive Histogram Equalization would improve low-contrast images.
* *Finding:* CLAHE across all images reduced mean PCK from 74.97% to 74.51% (clean dropped by -1.8 pp). Even when gated strictly on low-contrast images, it gave only +0.14 pp PCK (below our pre-established +0.3 pp adoption bar) and decreased APK (-0.04 pp).
* *Root Cause:* CLAHE operates on local $8 \times 8$ pixel contextual tiles. It artificially boosts micro-contrast across homogeneous regions, generating spurious gradient edges that corrupt HOG orientation histograms.

### 2. Why Contrast Normalization Doesn't Need Preprocessing:
* HOG includes explicit block normalisation: $\mathbf{f} = \mathbf{v} / \sqrt{\|\mathbf{v}\|^2 + \epsilon}$.
* Any global linear contrast degradation scales all pixel gradients in a block by constant $k$:
  $$\frac{k\mathbf{v}}{\sqrt{\|k\mathbf{v}\|^2 + \epsilon}} \approx \frac{\mathbf{v}}{\sqrt{\|\mathbf{v}\|^2}}$$
* The HOG descriptor is mathematically invariant to global affine illumination changes!

### 3. Denser HOG Scale Pyramid:
* Tested 15 levels per octave instead of 10.
* Result: +0.10 pp PCK, but -0.40 pp APK and doubled runtime. Rejected due to poor trade-off.

---

# 8. THE SCIENTIFIC PROTOCOL & VERSION HISTORY

### 8.1 The Split Protocol (How We Prevented Overfitting)
* Total PARSE test images = 205.
* **Tuning Split (Images 1–100):** Used exclusively to choose thresholds ($\sigma > 6$, sharpness $< 0.014$, brightness $< 0.40$, voting radius $r = 0.25$, temperature $T = 0.5$).
* **Held-Out Split (Images 101–205):** Frozen during design. Evaluated exactly once to report final numbers.
* **BUFFY:** Never touched during tuning; used strictly for zero-shot cross-dataset transfer validation.

### 8.2 Version History & The Single-Image Artifact
* **Version 1 (Mid-Review):** Noise and blur were detected but not acted on (`'High noise detected - unchanged'`). Metrics were flat or slightly negative.
* **Version 2 (Single-Image Artifact):** An earlier report claimed an impossible "+50 pp gain on noise" (14% $\rightarrow$ 64%). Upon scientific investigation, we uncovered that this file was generated by running code on a **single test image** (where per-joint accuracy can only be 0%, 50%, or 100%).
* **Version 3 (Current Validated Pipeline):** We withdrew the invalid single-image claim, implemented the rigorous 100/105 split protocol, introduced Immerkær's noise estimator, NLM, Richardson-Lucy, Flip TTA, and part-consistent voting, establishing fully verifiable and reproducible gains.

---

# 9. THE EXAMINER'S VIVA HITLIST: 25 TOUGH QUESTIONS & ANSWERS

### Q1. Why did you choose classical FMP instead of modern Deep Learning (YOLOv8, MediaPipe, OpenPose)?
**Answer:** "Deep learning models require millions of parameters, massive GPU compute, and thousands of annotated training images. They act as black boxes and fail unpredictably under out-of-distribution noise unless retrained with extensive data augmentation. Our project explores how far classical, mathematically interpretable vision (FMP) can be pushed under adverse imaging conditions *entirely at test-time on CPU*, without retraining or GPU infrastructure."

### Q2. How does Yang & Ramanan's FMP model avoid the exponential complexity of searching all possible human poses?
**Answer:** "The 26 body parts are connected as a tree-structured graph without closed loops. Because it is a tree, inference is solved exactly using dynamic programming (Viterbi message passing from leaf parts up to the root head). The spatial deformation cost between parent and child is a quadratic function, which allows the use of Felzenszwalb and Huttenlocher's Generalized Distance Transform to compute spatial message updates in $O(WH)$ linear time per scale instead of $O(W^2 H^2)$."

### Q3. Why does the model predict 26 parts when the benchmark evaluation only checks 14 joints?
**Answer:** "Yang & Ramanan showed that rigid 14-part models cannot handle foreshortening or non-rigid clothing deformations. By breaking limbs into smaller sub-parts (joints plus limb midpoints), each sub-part can independently choose from discrete mixture orientations. Once inference finds the optimal 26 parts, `PARSE_transback.m` extracts the 14 anatomical joints using a linear sparse projection matrix."

### Q4. How does Immerkær's noise estimator separate noise from image texture?
**Answer:** "Immerkær's estimator convolves the image with the kernel $M = [1, -2, 1; -2, 4, -2; 1, -2, 1]$. In continuous terms, this is the discrete approximation of the second-order Laplacian of the Laplacian. It has zero response to constant regions, linear ramps, and quadratic surfaces. Since genuine image structures are smooth or piecewise polynomial, the mask completely cancels the image structure and leaves only zero-mean uncorrelated Gaussian noise."

### Q5. Why did you divide by variance in your sharpness metric?
**Answer:** "A standard Laplacian variance $\text{Var}(\nabla^2 I)$ is sensitive to image contrast. A low-contrast image naturally has small gradients, so standard blur metrics falsely report low-contrast images as blurry. By computing $\text{Var}(\nabla^2 I_{\text{smooth}}) / \max(\text{Var}(I), 1)$, the metric becomes contrast-invariant."

### Q6. Why not apply all preprocessing filters simultaneously to every image?
**Answer:** "Because image filters have trade-offs. Deblurring sharpens edges but drastically amplifies noise. Denoising cleans noise but slightly softens subtle textures. Preprocessing must be gated so that an operation is applied only when its specific defect is detected. If an image is clean, passing it through unchanged protects against regression."

### Q7. Why did Richardson-Lucy deblurring work when Unsharp Masking failed?
**Answer:** "Unsharp masking is a linear high-pass filter that subtracts a blurred version from the original. It does not invert the blurring process; it merely boosts high-frequency noise and introduces ringing halos around edges. Richardson-Lucy is an iterative Bayesian maximum-likelihood deconvolution that explicitly inverts the Gaussian Point Spread Function under Poisson statistics, reconstructing true edge gradients."

### Q8. Why did Adaptive HOG fail to improve results without retraining?
**Answer:** "The FMP model uses linear SVM templates $w$ trained on the specific distribution of standard Felzenszwalb HOG features. When we modified the feature extraction code at test time (changing gradient weights), it caused a distribution shift: $w^T x_{\text{new}} \neq w^T x_{\text{orig}}$. The pre-trained filters no longer matched the new features. Modifying cell sizes or bins also alters vector dimensionality, making retraining mandatory."

### Q9. How does Flip Test-Time Augmentation (TTA) work in code?
**Answer:** "We run `detect_fast` on both the input image and its horizontal flip `im(:, end:-1:1, :)`. For the flipped detections, we reflect the horizontal coordinates: $x_1' = W + 1 - x_2$ and $x_2' = W + 1 - x_1$. Then we swap left and right part indices using `mirror = [1 2 15:26 3:14]` so left joints become right joints. Both candidate pools are merged."

### Q10. What is 'Part-Consistent' Pose Voting, and why did naive voting fail?
**Answer:** "Naive voting averages all joint coordinates from overlapping person detections. When candidates hypothesize different limb poses (e.g. one arm up vs. one arm down), naive voting averages them into an anatomically impossible position. In part-consistent voting, candidate $i$ votes for joint $j$ only if $\|P_j^{(i)} - P_j^*\| \le 0.25 \times \text{body\_size}$. This averages out small HOG grid quantization jitter while rejecting conflicting pose hypotheses."

### Q11. What anthropometric priors did you implement in `refine_pose.m`?
**Answer:** "We used Drillis & Contini (1966) and David Winter (2009) biomechanical proportions. We computed a robust torso scale anchored by a 30% total body height prior. We bounded limb segment lengths (upper arm, forearm, thigh, calf) relative to torso scale and enforced bilateral symmetry: $\text{target} = \text{median}(\text{left}, \text{right}, \text{expected})$. Joint outliers are projected back along their anatomical kinematic chains."

### Q12. Why did wrist accuracy dip slightly (-3.8 pp) on clean images?
**Answer:** "The wrist is at the end of the kinematic chain (shoulder $\rightarrow$ elbow $\rightarrow$ wrist) and has the smallest pixel extent, making it most vulnerable to HOG grid quantization. Our 95% paired-bootstrap confidence interval for wrist change is $[-9.5, +1.9]$, which includes zero. This indicates the dip is within normal statistical sample variance on 105 images."

### Q13. How do you know you didn't overfit the hyperparameters?
**Answer:** "Every threshold was tuned on PARSE test images 1–100. Images 101–205 were strictly held out until final evaluation. Furthermore, we evaluated the exact same pipeline on the BUFFY dataset (276 frames) without any tuning, achieving a +1.70 pp PCK and +2.12 pp APK gain."

### Q14. What are the 32 channels of Felzenszwalb HOG?
**Answer:** "They consist of:
* 18 contrast-sensitive orientation bins ($0^\circ - 360^\circ$),
* 9 contrast-insensitive orientation bins ($0^\circ - 180^\circ$),
* 4 block-normalised gradient energy features,
* 1 truncation feature indicating if the cell lies outside the image boundary."

### Q15. What is the difference between PCK and APK?
**Answer:** "PCK evaluates localization accuracy on the highest-scoring detection, measuring whether predicted joints fall within a distance threshold of ground truth. APK (Average Precision of Keypoints) evaluates detection and pose estimation jointly by ranking all candidates by confidence and calculating the area under the Precision-Recall curve."

### Q16. What is the difference between BUFFY and PARSE datasets?
**Answer:** "PARSE contains 205 full-body outdoor images annotated with 14 joints. BUFFY contains 276 frames from a TV show focusing on upper-body poses annotated with 10 joints (head, shoulders, elbows, wrists, hips). Our refinement module was evaluated on PARSE because it uses full-body proportions."

### Q17. How does the spring deformation model penalize displacement?
**Answer:** "The pairwise deformation term between parent $j$ and child $i$ is:
$$\psi(l_i, l_j) = w_{ij} \cdot [dx, dy, dx^2, dy^2]^T$$
Where $dx = x_i - x_j$ and $dy = y_i - y_j$. The quadratic terms $dx^2, dy^2$ act like Hooke's Law springs, penalizing deviations from the rest anchor offset."

### Q18. Why is global contrast enhancement (like stretching or CLAHE) unnecessary for HOG?
**Answer:** "HOG normalizes each $2 \times 2$ block by its L2 norm: $\mathbf{v} / \sqrt{\|\mathbf{v}\|^2 + \epsilon}$. If contrast drops uniformly, all gradient magnitudes are scaled by constant $c$. In the normalization formula, $c$ cancels in both numerator and denominator, making HOG inherently contrast-invariant."

### Q19. How did you compute statistical significance?
**Answer:** "We used paired-bootstrap hypothesis testing with 5,000 resamples. For each bootstrap iteration, we sampled images with replacement, computed the mean metric delta (Enhanced − Baseline), and derived the 2.5th and 97.5th percentiles to obtain the 95% Confidence Interval. If the interval excludes zero, the improvement is statistically significant."

### Q20. What was the '+50 pp on noise' error in Version 2?
**Answer:** "In early project reports, a checkpoint claimed PCK jumped from 14% to 64% on noisy images. When inspecting the cached MAT file, every per-joint accuracy was exactly 0, 50, or 100. The script had evaluated only a single image where 2 joints happened to be correct! We documented this error openly, discarded it, and established a reproducible evaluation on all 105 held-out images."

### Q21. What happens if an image has both noise and blur?
**Answer:** "Our decision ladder prioritizes noise removal first ($\sigma > 6$). Attempting deblurring on a noisy image amplifies noise uncontrollably because deconvolution acts as an inverse high-pass filter. Denoising first stabilizes the gradients."

### Q22. Could your pipeline be integrated with modern deep networks?
**Answer:** "Yes! Quality assessment, gated restoration, Flip TTA, and part-consistent pose voting are model-agnostic test-time wrappers. They can be placed around deep models like HRNet or OpenPose to improve robustness under harsh conditions without modifying network weights."

### Q23. Why is the runtime 2.4s instead of 1.3s?
**Answer:** "The original FMP takes $\approx 1.25\text{ s}$. Flip TTA runs inference twice (once on the original image, once on the mirrored image), which doubles detection time to $\approx 2.35\text{ s}$. Preprocessing takes only $0.13\text{ s}$, and voting and refinement take under $0.01\text{ s}$."

### Q24. How is the torso scale computed if hip or shoulder detections are wrong?
**Answer:** "We compute a prior torso scale as $0.30 \times \text{person\_height}$, where height is the bounding-box vertical extent. If the raw Euclidean distance between mid-shoulder and mid-hip falls outside $[0.18H, 0.45H]$, we discard the raw measurement and fall back to the 30% height prior."

### Q25. What is the single most important conclusion of your project?
**Answer:** "Classical gradient-based pose estimators do not need to be abandoned or retrained to achieve robustness under environmental degradation. By wrapping the unchanged model with no-reference quality assessment, targeted restoration, test-time flip voting, and anatomical refinement, we recover up to +6.8 pp PCK and +10.7 pp APK on degraded images while preserving clean accuracy."

---

# 10. QUICK-FIRE FORMULA CHEAT SHEET

1. **Immerkær Noise Estimator:**
   $$M = \begin{bmatrix} 1 & -2 & 1 \\ -2 & 4 & -2 \\ 1 & -2 & 1 \end{bmatrix}, \quad \sigma = \frac{\sum |I * M| \cdot \sqrt{\pi / 2}}{6(W-2)(H-2)}$$

2. **Contrast-Normalized Sharpness:**
   $$S = \frac{\text{Var}(\nabla^2 (I * G_{\sigma=0.8}))}{\max(\text{Var}(I), 1)}$$

3. **Gamma Correction:**
   $$I_{\text{out}} = I_{\text{in}}^\gamma \quad (\gamma = 0.67)$$

4. **HOG Block Normalisation:**
   $$\mathbf{f} = \frac{\mathbf{v}}{\sqrt{\|\mathbf{v}\|_2^2 + \epsilon}}$$

5. **Flip TTA Coordinate Reflection:**
   $$x_1' = W + 1 - x_2, \quad x_2' = W + 1 - x_1$$

6. **Part Mapping in PARSE:**
   $$\text{Mirror} = [1, 2, 15:26, 3:14]$$

7. **Pose Voting Softmax Weights:**
   $$w_i = \exp\left(\frac{s_i - s^*}{T}\right) \quad (T = 0.5)$$

8. **Part-Consistent Distance Gate:**
   $$\text{Weight} = w_i \cdot \mathbb{I}(\|P_j^{(i)} - P_j^*\| \le 0.25 \cdot \text{scale})$$

9. **PCK True Positive Condition:**
   $$\|\hat{p}_j - p_j\|_2 \le \alpha \cdot \max(W, H) \quad (\alpha = 0.1 \text{ for PARSE})$$

10. **Torso Height Prior:**
    $$\text{Torso Prior} = 0.30 \times (y_{\max} - y_{\min})$$

---

# 11. PPTX SLIDE-BY-SLIDE MASTER DECODER: EVERY METRIC, NUMBER & STATEMENT EXPLAINED

This section breaks down **all 20 slides** of your final review presentation (`Revisiting Classical Human Pose Estimation — Final Review.pptx`). When the examiner points at any number, percentage, or sentence on your screen, this is your exact script.

---

### SLIDE 1: Title & Overview
* **Title:** *"Revisiting Classical Human Pose Estimation"*
* **Subtitle:** *"Quality-gated restoration, flip test-time augmentation and pose voting around an unchanged Flexible Mixtures-of-Parts model"*
* **Team:** Group 1 · G O Ashwin Praveen · Atharva Unnikrishnan Nair · Karthik Kishor · Revathi V
* **What to Emphasize:** 
  * Notice the word **"unchanged"**. You did not retrain Yang & Ramanan's model; you did not change their weights or architecture. You built an intelligent **test-time wrapper** around it that makes it robust to environmental degradation.

---

### SLIDE 2: Why / The Motivation
* **Headline:** *"The challenge is not just finding a person"*
* **The Numbers (Baseline FMP PCK % on PARSE):**
  * **Clean:** `75.4%` (baseline accuracy on normal, good quality images)
  * **Dark:** `75.5%` (unaffected by underexposure because HOG gradients are still visible)
  * **Low Contrast:** `74.8%` (nearly identical to clean because HOG block normalisation scales it out)
  * **Noise:** `68.8%` (**A drop of −6.6 pp!**)
  * **Blur:** `68.9%` (**A drop of −6.5 pp!**)
* **What These Values Mean:**
  * While classical FMP is robust to lighting and contrast, it **collapses by ~6.6 percentage points** when subjected to high-frequency sensor noise or optical/defocus blur.
* **Why This Happens (Physical Reason):**
  * Noise introduces sharp, random pixel variations that generate **false gradient vectors** in arbitrary directions, corrupting HOG orientation bins.
  * Blur smooths out high frequencies, **flattening the true gradient boundaries** of arms and legs so part stencils fail to activate.

---

### SLIDE 3: Demo (Visual Outputs)
* **Statement:** *"TOP ROW: Baseline FMP on the degraded image. BOTTOM ROW: Full enhanced pipeline, same image."*
* **Statement:** *"NOTE: Largest gain per condition, chosen for illustration. The tables later are not cherry-picked."*
* **What to Tell the Examiner:**
  * *"These images show our pipeline fixing broken limb detections under blur and noise. We explicitly state on the slide that these visual examples are selected for qualitative illustration, but all our reported tables represent the rigorous average over all 105 held-out test images."*

---

### SLIDE 4: Pipeline (Baseline vs. Enhanced)
* **Baseline Flow (5 steps):** Image $\rightarrow$ HOG $\rightarrow$ Pretrained FMP $\rightarrow$ NMS $\rightarrow$ Pose
* **Enhanced Flow (7 steps):** Image $\rightarrow$ Quality analysis $\rightarrow$ Gated restoration $\rightarrow$ HOG $\rightarrow$ Pretrained FMP (image + mirror) $\rightarrow$ NMS + pose voting $\rightarrow$ Anatomical refinement $\rightarrow$ Pose
* **What to Tell the Examiner:**
  * Both rows use the **exact same pretrained FMP model and HOG feature parameters**. The comparison is strictly controlled.
  * Our novelty lies in **four test-time additions**:
    1. No-reference Quality Analysis (`image_quality.m`),
    2. Quality-Gated Restoration (`adaptive_preprocess.m`),
    3. Flip TTA + Part-Consistent Pose Voting (`detect_tta.m`, `pose_vote.m`),
    4. Anatomical Kinematic Refinement (`refine_pose.m`).

---

### SLIDE 5: Image Quality Analysis & Thresholds
* **The Decision Ladder Table:**
  | Condition | Signal Measured | Action Taken | Why This Operator? |
  |:---|:---|:---|:---|
  | **Noise** | $\sigma > 6.0$ | Non-Local Means (`imnlmfilt`) | Preserves repetitive limb edge structures while destroying zero-mean noise. |
  | **Blur** | Sharpness $< 0.014$ | Richardson-Lucy (`deconvlucy`) | Inverts the Gaussian blur PSF iteratively instead of just boosting noise like unsharp masking. |
  | **Dark** | Brightness $< 0.40$ | Gamma Correction ($\gamma = 0.67$) | Expands dark shadow regions without saturating bright highlights. |
  | **Low Contrast** | None | Passed through unchanged | HOG's L2 block normalization naturally cancels out uniform contrast drops. |
  | **Otherwise** | None | Passed through unchanged | Protects clean images from unnecessary filtering. |
* **Why These Exact Numbers?**
  * $\sigma > 6.0$: On PARSE test images, clean images scored $\sigma \le 9.2$, while noisy images scored $\sigma \ge 9.9$ (synthetic noise has $\sigma \approx 17$). A threshold of 6.0 is safely calibrated on the 100 tuning images to catch all degraded images without false triggers.
  * Sharpness $< 0.014$: Because sharpness is divided by image variance, low-contrast images stay above $0.014$, while genuinely blurred images drop below $0.010$.
* **Priority Rule (Noise $\rightarrow$ Blur $\rightarrow$ Dark):**
  * If an image is both noisy and blurry, running deconvolution first causes noise to blow up exponentially. Therefore, **noise must be cured first**.

---

### SLIDE 6: Features & The Adaptive HOG Study
* **Statement:** *"Edge directions per $4 \times 4$ pixel cell, normalised per block, so global contrast changes cancel out."*
* **The Numbers:**
  * **Weighted HOG:** $\pm 0.1$ pp PCK (no gain).
  * **Denser Scale Pyramid (15 vs. 10 levels/octave):** $+0.10$ pp PCK, but $-0.40$ pp APK, and doubled detection time.
* **What to Tell the Examiner:**
  * *"In our original abstract, we proposed Adaptive HOG. We implemented it in `code_basic/adaptive_features.cc` by weighting gradients based on image quality. However, our experiments proved a fundamental machine learning principle: **you cannot modify feature distributions at test time without retraining the linear classifier weights ($w^T x$)**. Because the FMP filters were trained on standard HOG, adapting features at test time causes a distribution shift, so accuracy stays flat ($\pm 0.1$ pp)."*

---

### SLIDE 7: Pose Model & Flip TTA
* **Key Concept:** Flexible Mixtures-of-Parts (26 parts, tree graph, dynamic programming).
* **Flip TTA Mechanism:**
  * Run FMP on $I$ (original) $\rightarrow$ candidate pool $A$.
  * Run FMP on $I_{\text{flip}} = I(:, \text{end}:-1:1, :)$ $\rightarrow$ candidate pool $B$.
  * For candidates in $B$, reflect horizontal coordinates: $x_1' = W + 1 - x_2$ and $x_2' = W + 1 - x_1$.
  * Swap bilateral parts using mapping `[1 2 15:26 3:14]` (left arm becomes right arm).
  * Pool the top 500 candidates from each pass together.

---

### SLIDE 8: Detection & Part-Consistent Voting
* **Statement:** *"NMS cleans up overlapping detections... On its own it does not improve joint coordinates."*
* **The Numbers & Parameters:**
  * $\text{IoU} > 0.5$: Candidate must overlap with the primary person detection.
  * Score $\ge s^* - 1.0$: Only consider candidates within $1.0$ log-score of the peak detection.
  * Softmax weight: $w_i = \exp((s_i - s^*) / 0.5)$.
  * **Distance gate $r \le 0.25 \times \text{body\_scale}$:** A candidate votes for joint $j$ only if its joint $j$ is within 25% of body size from the primary detection.
* **What This Value Gives:**
  * If candidate $A$ thinks the arm is raised and candidate $B$ thinks the arm is lowered, their distance is $> 0.25 \times \text{scale}$. The gate prevents them from averaging into an unphysical arm sticking out into empty space!
  * It raised tuning split mean PCK from $73.2\%$ to **$75.0\%$**.

---

### SLIDE 9: Evaluation Protocol
* **PCK Metric:** Correct if distance to ground truth $\le 0.1 \times \max(\text{Width}, \text{Height})$.
* **APK Metric:** Area under Precision-Recall curve across all detections.
* **Split Discipline:**
  * **Tuning split (Images 1–100):** Every threshold, parameter, and operator was chosen here.
  * **Held-out split (Images 101–205, N = 105):** Never used during design; evaluated only for the final report.
* **Paired Bootstrap:** 5,000 resamples to compute 95% Confidence Intervals.
* **Runtime:** Baseline $\approx 1.3\text{ s} \rightarrow$ Full Pipeline $\approx 2.4\text{ s}$ per image on CPU.

---

### SLIDE 10: Clean-Set Results (The "Honest Science" Slide)
* **The Table:**
  | Metric | Baseline FMP | Full Pipeline | Change ($\Delta$) | 95% Bootstrap CI |
  |:---|:---:|:---:|:---:|:---:|
  | **Mean PCK** | 75.37% | 76.53% | **+1.16 pp** | **[−1.53, +3.61]** |
  | **Mean APK** | 66.19% | 68.07% | **+1.88 pp** | — |
* **What Every Value Means:**
  * `+1.16 pp`: On clean images, accuracy went up from 75.37% to 76.53%.
  * `[-1.53, +3.61]`: In 95% of bootstrap resamples, the true change lies between $-1.53$ and $+3.61$.
  * **Crucial Viva Explanation:** Because the lower bound is negative ($-1.53$), **the interval crosses zero**. In statistics, this means the gain is **not statistically significant** at the 95% confidence level.
  * **Why This Impresses Examiners:** You are not claiming a false victory on clean images. You openly state: *"Our pipeline is designed for degraded images; on clean images, it satisfies the 'first do no harm' principle."*

---

### SLIDE 11: Per-Joint Breakdown (PARSE & BUFFY)

#### PARSE Clean Held-Out PCK % (105 Images):
| Joint | Baseline | Full | Change ($\Delta$) | What Happened? |
|:---|:---:|:---:|:---:|:---|
| **Head** | 91.9% | 90.5% | −1.4 pp | Minor sample variation; CI includes zero. |
| **Shoulder** | 88.6% | 87.1% | −1.5 pp | Minor sample variation; CI includes zero. |
| **Elbow** | 68.6% | **70.5%** | **+1.9 pp** | Improved by symmetry prior. |
| **Wrist** | 51.4% | 47.6% | −3.8 pp | End of kinematic chain, high rotational freedom. CI $[-9.5, +1.9]$ includes zero. |
| **Hip** | 80.0% | **83.3%** | **+3.3 pp** | Stabilized by torso scale prior. |
| **Knee** | 77.6% | **81.4%** | **+3.8 pp** | Constrained by thigh bone bounds. |
| **Ankle** | 69.5% | **75.2%** | **+5.7 pp** | **Statistically significant gain!** CI $[+1.4, +10.5]$ strictly excludes zero. |
| **Mean** | 75.4% | **76.5%** | **+1.2 pp** | Overall accuracy improved. |

#### BUFFY Cross-Dataset PCK % (276 TV Frames, Zero-Shot):
* **Head:** 96.0% $\rightarrow$ 95.7% (−0.4)
* **Shoulder:** 97.1% $\rightarrow$ 97.1% (0.0)
* **Elbow:** 87.5% $\rightarrow$ **90.6%** (+3.1)
* **Wrist:** 67.2% $\rightarrow$ **68.8%** (+1.6)
* **Hip:** 85.3% $\rightarrow$ **89.5%** (+4.2)
* **Average PCK:** 86.63% $\rightarrow$ **88.33% (+1.70 pp)**
* **Average APK:** 76.63% $\rightarrow$ **78.75% (+2.12 pp)**
* **What to Tell the Examiner:**
  * *"BUFFY consists of real video frames from a television show. We did not tune a single parameter on BUFFY. The fact that PCK rose by +1.70 pp and APK rose by +2.12 pp proves that our quality gates and voting generalise across different camera sensors and environments."*

---

### SLIDE 12: Preprocessing Activity (Share of Images Restored)
* **The Numbers:**
  * **Noise:** `100%` (105 / 105 images triggered NLM)
  * **Blur:** `89%` (93 / 105 images triggered Richardson-Lucy)
  * **Dark:** `69%` (72 / 105 images triggered Gamma correction)
  * **Low Contrast:** `15%` (16 / 105 images triggered a fix because they were also slightly dark/noisy)
  * **Clean:** `31%` (33 / 105 images triggered a fix)
* **What These Values Prove:**
  * **100% on Noise** shows the Immerkær noise estimator has a 100% detection rate on synthetic noise.
  * **31% on Clean** shows that real-world outdoor photos naturally possess slight film grain, soft focus, or dim lighting. Even though 31% were filtered, clean accuracy did not degrade, proving our filters are gentle and safe.

---

### SLIDE 13: Robustness Table (THE HEADLINE SLIDE)

| Condition | Baseline PCK | Full Pipeline PCK | $\Delta$ PCK [95% CI] | Baseline APK | Full APK | $\Delta$ APK |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Dark** | 75.51% | **78.37%** | **+2.86 pp [+0.54, +5.03]** | 67.13% | **69.30%** | +2.17 pp |
| **Low Contrast** | 74.76% | **76.80%** | +2.04 pp [−0.48, +4.22] | 66.28% | **69.85%** | +3.57 pp |
| **Noise** | 68.78% | **74.76%** | **+5.99 pp [+2.69, +9.12]** | 57.26% | **65.38%** | **+8.12 pp** |
| **Blur** | 68.91% | **75.71%** | **+6.80 pp [+3.95, +10.07]** | 57.43% | **68.14%** | **+10.71 pp** |

#### What to Explain Out Loud for Each Row:
1. **Noise (+5.99 pp PCK, +8.12 pp APK):**
   * Baseline collapsed from 75.37% down to 68.78%.
   * Our pipeline brings it back up to 74.76%, **recovering ~91% of the lost accuracy**.
   * The 95% CI $[+2.69, +9.12]$ is strictly positive $\rightarrow$ **Statistically significant**.
2. **Blur (+6.80 pp PCK, +10.71 pp APK):**
   * Baseline collapsed to 68.91%.
   * Our pipeline brings it up to 75.71%, which is **higher than the clean baseline (75.37%)**!
   * The 95% CI $[+3.95, +10.07]$ is strictly positive $\rightarrow$ **Statistically significant**.
3. **Dark (+2.86 pp PCK, +2.17 pp APK):**
   * Gamma correction expands shadow gradients, boosting PCK significantly ($[+0.54, +5.03]$).
4. **Low Contrast (+2.04 pp PCK, +3.57 pp APK):**
   * The gain comes primarily from Flip TTA and pose voting, not preprocessing (since low contrast is bypassed).

---

### SLIDE 14, 15, 16: Model Outputs (Visual Validation)
* **Images Shown:** PARSE images `109`, `146`, `174`, `182`.
* **Statement:** *"Four held-out images drawn at random with a fixed seed. Not picked by result."*
* **What to Tell the Examiner:**
  * *"In computer vision papers, authors often cherry-pick only their best detections. We avoided this bias by selecting test images using a pseudo-random seed before running inference, ensuring honest qualitative representation."*

---

### SLIDE 17: Takeaways
1. **Clean Images:** Do no harm (+1.2 PCK, +1.9 APK).
2. **Dark & Low Contrast:** HOG already normalizes contrast; gamma + voting gives +2.9 PCK on dark images.
3. **Noise & Blur:** The right operator matters (NLM for noise, Richardson-Lucy for blur give +6.0 and +6.8 PCK).
4. **One Rule is Not Enough:** Measure each image, apply at most one fix, and test on held-out images.

---

### SLIDE 18 & 19: Roadmap & Abstract Check
* **CLAHE:** Tested $\rightarrow$ Reduced PCK by $-0.5$ on clean images and gave only $+0.1$ on low contrast $\rightarrow$ **Rejected**.
* **Adaptive HOG:** Tested $\rightarrow$ Feature distribution shift without retraining gave $\pm 0.1$ $\rightarrow$ **Reported as negative finding**.
* **Denoising & Deblurring:** NLM beat bilateral filter; Richardson-Lucy beat unsharp masking $\rightarrow$ **Delivered**.
* **Pose Voting & Refinement:** Part-consistent voting and biomechanical bounds $\rightarrow$ **Delivered**.
* **Efficiency:** $\approx 1.3\text{ s} \rightarrow 2.4\text{ s}$ on CPU, no GPU $\rightarrow$ **Delivered**.

---

### SLIDE 20: Conclusion
* **Summary Statement:**
  > *"We have demonstrated that classical Flexible Mixtures-of-Parts pose estimation does not need to be discarded in challenging environments. By surrounding the unchanged model with automated image-quality gating, test-time flip voting, and anthropometric constraints, we recover almost all lost accuracy under noise and blur while maintaining real-time CPU performance."*

---
*Created for Group 1 Viva Preparation — Atharva Nair, Ashwin Praveen, Karthik Kishor, Revathi V.*

