function demo_complete(imgIndex, degradationCondition)
% DEMO_COMPLETE
% Interactive demonstration for:
% "Improving Classical Human Pose Estimation Using Image Preprocessing and Adaptive Feature Extraction"
%
% Usage:
%   demo_complete()                    - runs on default image (clean im0101.jpg)
%   demo_complete(1, 'dark')           - runs on dark degraded image
%   demo_complete(5, 'noise')          - runs on noisy image
%   demo_complete(10, 'lowcontrast')   - runs on low-contrast image
%   demo_complete(3, 'blur')           - runs on blurred image
%   demo_complete(2, 'clean')          - runs on clean test image

    if nargin < 1 || isempty(imgIndex)
        imgIndex = 1;
    end
    if nargin < 2 || isempty(degradationCondition)
        degradationCondition = 'dark';
    end

    clc;
    fprintf('========================================================================\n');
    fprintf('  INTERACTIVE DEMO: ADAPTIVE FMP POSE ESTIMATION & REFINEMENT\n');
    fprintf('========================================================================\n\n');

    codeFullDir = fileparts(mfilename('fullpath'));
    rootDir     = fileparts(codeFullDir);
    codeBasicDir = fullfile(rootDir, 'code_basic');

    addpath(codeFullDir, '-begin');
    addpath(fullfile(codeFullDir, 'detection'), '-begin');
    addpath(fullfile(codeFullDir, 'visualization'), '-begin');
    addpath(fullfile(codeFullDir, 'evaluation'), '-begin');
    globals;

    % Load model & ground truth
    fprintf('Loading PARSE model and test annotations...\n');
    load(fullfile(codeBasicDir, 'PARSE_model.mat'), 'model');
    model.thresh = min(model.thresh, -1);
    [~, test] = PARSE_data('PARSE');

    % Load image
    imgNum = 100 + imgIndex;
    if strcmpi(degradationCondition, 'clean')
        imgPath = fullfile(codeFullDir, 'data', 'PARSE', sprintf('im%04d.jpg', imgNum));
    else
        imgPath = fullfile(codeFullDir, 'data', 'PARSE_degraded', lower(degradationCondition), sprintf('im%04d.jpg', imgNum));
    end

    if ~exist(imgPath, 'file')
        error('Image not found: %s', imgPath);
    end

    im = imread(imgPath);
    gtPoints = test(imgIndex).obj(1).point;

    fprintf('Input image : %s (Condition: %s)\n', sprintf('im%04d.jpg', imgNum), upper(degradationCondition));

    % 1. Image Quality Analysis & Adaptive Preprocessing
    fprintf('\n[Step 1 & 2] Performing Image Quality Analysis & Adaptive Preprocessing...\n');
    [imEnhanced, qualityInfo] = adaptive_preprocess(im);
    fprintf('  - Brightness: %.3f\n', qualityInfo.brightness);
    fprintf('  - Contrast  : %.3f\n', qualityInfo.contrast);
    fprintf('  - Noise     : %.4f\n', qualityInfo.noise);
    fprintf('  - Sharpness : %.5f\n', qualityInfo.sharpness);
    if isempty(qualityInfo.operations)
        fprintf('  - Operations: None needed (Image is optimal quality)\n');
    else
        fprintf('  - Operations: %s\n', strjoin(qualityInfo.operations, ', '));
    end

    % 2. Baseline FMP Detection
    fprintf('\n[Step 3A] Running Baseline FMP Inference...\n');
    tic;
    boxesBase = detect_fast(im, model, model.thresh);
    boxesBase = nms(boxesBase, 0.3);
    timeBase = toc;
    poseBase = box2obj(bestoverlap(boxesBase, test(imgIndex).obj(1), 0.3));
    detBase14 = PARSE_transback(struct('obj', poseBase));
    ptsBase = [];
    if ~isempty(detBase14.obj), ptsBase = detBase14.obj(1).point; end
    fprintf('  - Baseline detected in %.2f s\n', timeBase);

    % 3. Adaptive Preprocessing + FMP with flip TTA and pose voting
    fprintf('\n[Step 3B] Running Enhanced Pipeline (Preprocessing + Flip TTA + Pose Voting)...\n');
    tic;
    boxesComb = enhanced_detect(im, model);
    timeComb = toc;
    poseComb = [];
    bestComb = bestoverlap(boxesComb, test(imgIndex).obj(1), 0.3);
    if ~isempty(bestComb), poseComb = pose_vote(bestComb, boxesComb); end
    detComb14 = PARSE_transback(struct('obj', poseComb));
    ptsComb = [];
    if ~isempty(detComb14.obj), ptsComb = detComb14.obj(1).point; end
    fprintf('  - Enhanced pipeline detected in %.2f s\n', timeComb);

    % 4. Anatomical Pose Refinement
    fprintf('\n[Step 4] Applying Anatomical Pose Refinement (Geometric & Symmetry Constraints)...\n');
    [detRef14, numCorr] = apply_pose_refinement(detComb14);
    ptsRef = [];
    if ~isempty(detRef14.obj), ptsRef = detRef14.obj(1).point; end
    fprintf('  - Refinement applied (%d joint constraints enforced)\n', numCorr);

    % 5. Error Metrics to Ground Truth
    if ~isempty(ptsBase)
        errBase = mean(sqrt(sum((ptsBase - gtPoints).^2, 2)));
        fprintf('\nMean Euclidean Keypoint Error to Ground Truth:\n');
        fprintf('  - Baseline FMP Error : %.2f pixels\n', errBase);
    end
    if ~isempty(ptsComb)
        errComb = mean(sqrt(sum((ptsComb - gtPoints).^2, 2)));
        fprintf('  - Enhanced FMP Error : %.2f pixels (Delta: %+.2f px)\n', errComb, errBase - errComb);
    end
    if ~isempty(ptsRef)
        errRef = mean(sqrt(sum((ptsRef - gtPoints).^2, 2)));
        fprintf('  - Refined Pose Error : %.2f pixels (Delta: %+.2f px)\n', errRef, errBase - errRef);
    end

    % 6. Visualization
    f = figure('Name', sprintf('Pose Estimation Demo - %s (%s)', sprintf('im%04d.jpg', imgNum), degradationCondition), ...
               'Position', [50 100 1350 420]);

    subplot(1, 4, 1);
    imagesc(im); axis image; axis off;
    title(sprintf('(1) Input (%s)', upper(degradationCondition)), 'FontSize', 11, 'FontWeight', 'bold');

    subplot(1, 4, 2);
    plot_parse_skeleton(im, ptsBase, struct('lineWidth', 3, 'markerSize', 5, 'title', '(2) Baseline FMP'));

    subplot(1, 4, 3);
    plot_parse_skeleton(imEnhanced, ptsComb, struct('lineWidth', 3, 'markerSize', 5, 'title', '(3) Enhanced FMP'));

    subplot(1, 4, 4);
    plot_parse_skeleton(im, ptsRef, struct('lineWidth', 3, 'markerSize', 5, 'title', '(4) Refined (Ours)'));

    fprintf('\nDemo visualization figure opened successfully.\n');
end
