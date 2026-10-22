# Adaptive FMP Training Patch

## Why this is needed

The current project evaluates `adaptive_features` at test time, but the supplied `PARSE_model.mat` was trained using the repository's original `features` HOG extractor.

Both extractors output 32 channels, but `adaptive_features` changes gradient magnitudes for low-contrast, noisy, and blurred images before HOG normalization. Therefore the trained filters are not guaranteed to be optimal for the adaptive feature distribution.

This patch fine-tunes the pretrained FMP model using the same `adaptive_features`/`adaptive_featpyramid` pipeline used during enhanced inference.

## Files

- `code_full/learning/adaptive_learning_detect.m`
  - Copy of the original learning detector with `adaptive_featpyramid()` instead of `featpyramid()`.
- `code_full/learning/adaptive_train.m`
  - Copy of the original structured-SVM training routine that routes positive/negative mining through `adaptive_learning_detect()`.
- `code_full/make_adaptive_training_set.m`
  - Builds 500 positive examples: 100 clean PARSE training images + 100 each of dark, low-contrast, noisy and blurred versions.
  - Uses only PARSE frames 1:100, so the 101:305 test set is not used for training.
- `code_full/train_adaptive_fmp.m`
  - Fine-tunes the supplied pretrained FMP model and saves a new model without overwriting the baseline.
- `code_full/evaluate_trained_adaptive.m`
  - Compares baseline FMP against the newly trained adaptive-HOG model on clean PARSE and degraded PARSE images.

## Required data

The ZIP contains only a small portion of the INRIA negative dataset. The original FMP training code expects the full INRIA training negatives (frames 615:1832). If those files are missing on the machine where MATLAB is run, the training script will stop rather than silently train with an incomplete negative set.

The PARSE training/test images already present in the project are sufficient for the positive side.

## MATLAB steps

1. Open MATLAB.
2. Move to:

   `human-pose-estimation-master - enhanced/code_full`

3. Make sure `code_basic/adaptive_features.mexw64` exists. If not, compile the adaptive MEX source for the local MATLAB/compiler setup.
4. Run the robust fine-tuning experiment:

   `model = train_adaptive_fmp(true, 1);`

5. Evaluate it first on 25 degraded images per condition:

   `results = evaluate_trained_adaptive(fullfile('..','code_basic','PARSE_adaptive_robust_model.mat'), 25);`

6. If the result is promising, run all 205 degraded images per condition:

   `results = evaluate_trained_adaptive(fullfile('..','code_basic','PARSE_adaptive_robust_model.mat'), 205);`

## Recommended scientific comparison

Run two fine-tuning variants:

### A. Compatibility-only model

```matlab
model = train_adaptive_fmp(false, 1);
```

This uses the clean PARSE training frames only. It tests whether simply training the FMP filters on the adaptive feature representation fixes the feature/model mismatch.

### B. Robust adaptive model

```matlab
model = train_adaptive_fmp(true, 1);
```

This adds controlled dark/low-contrast/noise/blur training examples. It tests whether adaptive HOG plus robustness-oriented training improves adverse-condition performance.

These two experiments should be kept separate in the report. Otherwise an improvement could not be attributed cleanly to feature compatibility versus degradation augmentation.

## Do not overwrite the baseline

The original `code_basic/PARSE_model.mat` is never modified. New models are saved as:

- `PARSE_adaptive_clean_model.mat`
- `PARSE_adaptive_robust_model.mat`

## Important limitation

This patch addresses the first major issue we identified. It does not claim that adaptive HOG must improve the final metric. The current adaptive weighting is deliberately small and is applied before HOG normalization, so some of its effect can be reduced by normalization. The experiment will tell us whether model retraining is sufficient; if not, the next step should be redesigning the adaptive feature weighting rather than tuning thresholds blindly.
