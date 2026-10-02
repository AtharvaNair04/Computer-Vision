function results = run_complete_experiments(numDegradedImages)
% RUN_COMPLETE_EXPERIMENTS
% End-to-end evaluation and figure generation for:
% "Improving Classical Human Pose Estimation Using Image Preprocessing and Adaptive Feature Extraction"
%
% Evaluates:
%   1. Clean PARSE test set (all 205 images)
%   2. Degraded PARSE test sets (Dark, Low-Contrast, Noise, Blur)
%
% Compares 5 Model Configurations:
%   (1) Baseline FMP (Yang & Ramanan, CVPR 2011)
%   (2) Adaptive Preprocessing + FMP
%   (3) Adaptive HOG Feature Extraction + FMP
%   (4) Combined (Adaptive Preprocessing + Adaptive HOG) + FMP
%   (5) Full Pipeline with Pose Refinement (Ours)
%
% Computes:
%   - Official APK (Average Precision of Keypoints, Yang & Ramanan protocol)
%   - Official PCK (Percentage of Correct Keypoints, Yang & Ramanan protocol)
%   - 7 Anatomical Keypoint Breakdowns (Head, Shoulder, Elbow, Wrist, Hip, Knee, Ankle)
%   - PCK vs Distance Threshold Curves
%   - Saves publication figures to figures/ directory

    if nargin < 1 || isempty(numDegradedImages)
        numDegradedImages = 40; % Representative test set for fast, thorough evaluation
    end

    clc;
    fprintf('========================================================================\n');
    fprintf('  HUMAN POSE ESTIMATION: COMPREHENSIVE EXPERIMENTAL EVALUATION SUITE\n');
    fprintf('  Project: Image Preprocessing + Adaptive Features + Pose Refinement\n');
    fprintf('========================================================================\n\n');

    %% -------------------------------------------------------------
    % Setup Paths
    % -------------------------------------------------------------
    codeFullDir  = fileparts(mfilename('fullpath'));
    rootDir      = fileparts(codeFullDir);
    codeBasicDir = fullfile(rootDir, 'code_basic');
    figuresDir   = fullfile(rootDir, 'figures');
    if ~exist(figuresDir, 'dir'), mkdir(figuresDir); end

    addpath(codeFullDir, '-begin');
    addpath(fullfile(codeFullDir, 'detection'), '-begin');
    addpath(fullfile(codeFullDir, 'evaluation'), '-begin');
    addpath(fullfile(codeFullDir, 'visualization'), '-begin');
    addpath(codeBasicDir);
    addpath(fullfile(codeBasicDir, 'mex_pc'));
    addpath(fullfile(codeFullDir, 'mex_pc'));

    globals;

    %% -------------------------------------------------------------
    % Load Model & PARSE Test Data
    % -------------------------------------------------------------
    fprintf('Loading PARSE model and test data...\n');
    load(fullfile(codeBasicDir, 'PARSE_model.mat'), 'model');
    model.thresh = min(model.thresh, -1);
    [~, test] = PARSE_data('PARSE');
    numTotalTest = length(test);
    fprintf('Model threshold: %.4f | Total PARSE test images: %d\n\n', model.thresh, numTotalTest);

    mirror = [6 5 4 3 2 1 12 11 10 9 8 7 14 13];
    order7 = [14 9 8 7 3 2 1]; % Head, Shoulder, Elbow, Wrist, Hip, Knee, Ankle
    jointNames7 = {'Head', 'Shoulder', 'Elbow', 'Wrist', 'Hip', 'Knee', 'Ankle'};
    thresholds = 0.02:0.02:0.26;

    results = struct();

    %% =============================================================
    % PART 1: CLEAN PARSE BENCHMARK EVALUATION (All 205 Test Images)
    % =============================================================
    fprintf('========================================================================\n');
    fprintf(' PART 1: CLEAN PARSE BENCHMARK (All 205 Images)\n');
    fprintf('========================================================================\n');

    cleanCacheBase  = fullfile(codeFullDir, 'cache', 'PARSE_baseline_eval_detections_adaptive_comparison.mat');
    cleanCacheAdap  = fullfile(codeFullDir, 'cache', 'PARSE_adaptive_eval_adaptive_detections_adaptive_comparison.mat');
    cleanCacheComb  = fullfile(codeFullDir, 'cache', 'PARSE_combined_detections.mat');

    dataBase = load(cleanCacheBase);
    dataAdap = load(cleanCacheAdap);
    dataComb = load(cleanCacheComb);

    % Map to 14 PARSE joints
    detBase14 = PARSE_transback(dataBase.det);
    poseBase14 = PARSE_transback(dataBase.pose);

    detAdap14 = PARSE_transback(dataAdap.det);
    poseAdap14 = PARSE_transback(dataAdap.pose);

    detComb14 = PARSE_transback(dataComb.det);
    poseComb14 = PARSE_transback(dataComb.pose);

    % Apply Pose Refinement on Combined
    fprintf('Applying Pose Refinement (Box 5) on Clean Test set...\n');
    [detRef14, ~]  = apply_pose_refinement(detComb14);
    [poseRef14, ~] = apply_pose_refinement(poseComb14);

    % Compute Clean Metrics
    % 1. Baseline
    results.clean.baseline.apk_all  = eval_apk(detBase14, test, 0.1);
    results.clean.baseline.pck_all  = eval_pck(poseBase14, test, 0.1);
    results.clean.baseline.apk7     = map_to_7(results.clean.baseline.apk_all, mirror, order7);
    results.clean.baseline.pck7     = map_to_7(results.clean.baseline.pck_all, mirror, order7);
    results.clean.baseline.meanAPK  = mean(results.clean.baseline.apk7);
    results.clean.baseline.meanPCK  = mean(results.clean.baseline.pck7);
    results.clean.baseline.pckCurve = compute_pck_curve(poseBase14, test, thresholds, mirror, order7);

    % 2. Adaptive HOG
    results.clean.adaptiveHOG.apk_all  = eval_apk(detAdap14, test, 0.1);
    results.clean.adaptiveHOG.pck_all  = eval_pck(poseAdap14, test, 0.1);
    results.clean.adaptiveHOG.apk7     = map_to_7(results.clean.adaptiveHOG.apk_all, mirror, order7);
    results.clean.adaptiveHOG.pck7     = map_to_7(results.clean.adaptiveHOG.pck_all, mirror, order7);
    results.clean.adaptiveHOG.meanAPK  = mean(results.clean.adaptiveHOG.apk7);
    results.clean.adaptiveHOG.meanPCK  = mean(results.clean.adaptiveHOG.pck7);
    results.clean.adaptiveHOG.pckCurve = compute_pck_curve(poseAdap14, test, thresholds, mirror, order7);

    % 3. Combined
    results.clean.combined.apk_all  = eval_apk(detComb14, test, 0.1);
    results.clean.combined.pck_all  = eval_pck(poseComb14, test, 0.1);
    results.clean.combined.apk7     = map_to_7(results.clean.combined.apk_all, mirror, order7);
    results.clean.combined.pck7     = map_to_7(results.clean.combined.pck_all, mirror, order7);
    results.clean.combined.meanAPK  = mean(results.clean.combined.apk7);
    results.clean.combined.meanPCK  = mean(results.clean.combined.pck7);
    results.clean.combined.pckCurve = compute_pck_curve(poseComb14, test, thresholds, mirror, order7);

    % 4. Refined
    results.clean.refined.apk_all  = eval_apk(detRef14, test, 0.1);
    results.clean.refined.pck_all  = eval_pck(poseRef14, test, 0.1);
    results.clean.refined.apk7     = map_to_7(results.clean.refined.apk_all, mirror, order7);
    results.clean.refined.pck7     = map_to_7(results.clean.refined.pck_all, mirror, order7);
    results.clean.refined.meanAPK  = mean(results.clean.refined.apk7);
    results.clean.refined.meanPCK  = mean(results.clean.refined.pck7);
    results.clean.refined.pckCurve = compute_pck_curve(poseRef14, test, thresholds, mirror, order7);

    print_clean_summary(results.clean, jointNames7);

    %% =============================================================
    % PART 2: DEGRADED ROBUSTNESS EVALUATION (Dark, Contrast, Noise, Blur)
    % =============================================================
    fprintf('\n========================================================================\n');
    fprintf(' PART 2: ROBUSTNESS EVALUATION UNDER ADVERSE CONDITIONS\n');
    fprintf(' Evaluating %d test images per degradation condition...\n', numDegradedImages);
    fprintf('========================================================================\n');

    conditions = {'dark', 'lowcontrast', 'noise', 'blur'};
    degradedDir = fullfile(codeFullDir, 'data', 'PARSE_degraded');
    testSub = test(1:numDegradedImages);

    for c = 1:length(conditions)
        cond = conditions{c};
        fprintf('\n>>> Condition [%d/%d]: %s <<<\n', c, length(conditions), upper(cond));
        condImageDir = fullfile(degradedDir, cond);

        cacheCondFile = fullfile(codeFullDir, 'cache', sprintf('degraded_eval_%s_N%d.mat', cond, numDegradedImages));

        if exist(cacheCondFile, 'file')
            fprintf('  Loading cached degraded detections from: %s\n', cacheCondFile);
            cachedCond = load(cacheCondFile);
            detBaseRaw = cachedCond.detBaseRaw;
            detPreRaw  = cachedCond.detPreRaw;
            detAdapRaw = cachedCond.detAdapRaw;
            detCombRaw = cachedCond.detCombRaw;
            poseBaseRaw = cachedCond.poseBaseRaw;
            posePreRaw  = cachedCond.posePreRaw;
            poseAdapRaw = cachedCond.poseAdapRaw;
            poseCombRaw = cachedCond.poseCombRaw;
        else
            % Allocate storage
            detBaseRaw  = repmat(struct('obj', []), 1, numDegradedImages);
            detPreRaw   = repmat(struct('obj', []), 1, numDegradedImages);
            detAdapRaw  = repmat(struct('obj', []), 1, numDegradedImages);
            detCombRaw  = repmat(struct('obj', []), 1, numDegradedImages);

            poseBaseRaw = repmat(struct('obj', []), 1, numDegradedImages);
            posePreRaw  = repmat(struct('obj', []), 1, numDegradedImages);
            poseAdapRaw = repmat(struct('obj', []), 1, numDegradedImages);
            poseCombRaw = repmat(struct('obj', []), 1, numDegradedImages);

            ticCond = tic;
            for n = 1:numDegradedImages
                imgNum = 100 + n;
                imgFile = fullfile(condImageDir, sprintf('im%04d.jpg', imgNum));
                imDeg = imread(imgFile);
                gtObj = testSub(n).obj;

                % (1) Baseline FMP
                bBase = detect_fast(imDeg, model, model.thresh);
                poseBaseRaw(n).obj = extract_pose(bBase, gtObj);
                bBase = nms(bBase, 0.3);
                detBaseRaw(n).obj = box2obj(bBase);

                % (2) Adaptive Preprocessing + FMP
                [imPre, ~] = adaptive_preprocess(imDeg);
                bPre = detect_fast(imPre, model, model.thresh);
                posePreRaw(n).obj = extract_pose(bPre, gtObj);
                bPre = nms(bPre, 0.3);
                detPreRaw(n).obj = box2obj(bPre);

                % (3) Adaptive HOG + FMP
                bAdap = adaptive_detect_fast(imDeg, model, model.thresh);
                poseAdapRaw(n).obj = extract_pose(bAdap, gtObj);
                bAdap = nms(bAdap, 0.3);
                detAdapRaw(n).obj = box2obj(bAdap);

                % (4) Combined (Adaptive Preprocessing + Adaptive HOG)
                bComb = adaptive_detect_fast(imPre, model, model.thresh);
                poseCombRaw(n).obj = extract_pose(bComb, gtObj);
                bComb = nms(bComb, 0.3);
                detCombRaw(n).obj = box2obj(bComb);

                if mod(n, 10) == 0 || n == numDegradedImages
                    fprintf('    [%s] Processed %d/%d images (elapsed: %.1fs)\n', ...
                        cond, n, numDegradedImages, toc(ticCond));
                end
            end

            save(cacheCondFile, 'detBaseRaw', 'detPreRaw', 'detAdapRaw', 'detCombRaw', ...
                                'poseBaseRaw', 'posePreRaw', 'poseAdapRaw', 'poseCombRaw');
            fprintf('  Saved detections to cache: %s\n', cacheCondFile);
        end

        % Map to 14 PARSE joints
        detBase14 = PARSE_transback(detBaseRaw);
        detPre14  = PARSE_transback(detPreRaw);
        detAdap14 = PARSE_transback(detAdapRaw);
        detComb14 = PARSE_transback(detCombRaw);

        poseBase14 = PARSE_transback(poseBaseRaw);
        posePre14  = PARSE_transback(posePreRaw);
        poseAdap14 = PARSE_transback(poseAdapRaw);
        poseComb14 = PARSE_transback(poseCombRaw);

        % (5) Full Pipeline with Pose Refinement (Box 5)
        [detRef14, ~]  = apply_pose_refinement(detComb14);
        [poseRef14, ~] = apply_pose_refinement(poseComb14);

        % Calculate metrics for all 5 models under this condition
        condRes = struct();

        % Baseline
        condRes.baseline.apk7 = map_to_7(eval_apk(detBase14, testSub, 0.1), mirror, order7);
        condRes.baseline.pck7 = map_to_7(eval_pck(poseBase14, testSub, 0.1), mirror, order7);
        condRes.baseline.meanAPK = mean(condRes.baseline.apk7);
        condRes.baseline.meanPCK = mean(condRes.baseline.pck7);
        condRes.baseline.pckCurve = compute_pck_curve(poseBase14, testSub, thresholds, mirror, order7);

        % Preprocessing alone
        condRes.preprocess.apk7 = map_to_7(eval_apk(detPre14, testSub, 0.1), mirror, order7);
        condRes.preprocess.pck7 = map_to_7(eval_pck(posePre14, testSub, 0.1), mirror, order7);
        condRes.preprocess.meanAPK = mean(condRes.preprocess.apk7);
        condRes.preprocess.meanPCK = mean(condRes.preprocess.pck7);
        condRes.preprocess.pckCurve = compute_pck_curve(posePre14, testSub, thresholds, mirror, order7);

        % Adaptive HOG alone
        condRes.adaptiveHOG.apk7 = map_to_7(eval_apk(detAdap14, testSub, 0.1), mirror, order7);
        condRes.adaptiveHOG.pck7 = map_to_7(eval_pck(poseAdap14, testSub, 0.1), mirror, order7);
        condRes.adaptiveHOG.meanAPK = mean(condRes.adaptiveHOG.apk7);
        condRes.adaptiveHOG.meanPCK = mean(condRes.adaptiveHOG.pck7);
        condRes.adaptiveHOG.pckCurve = compute_pck_curve(poseAdap14, testSub, thresholds, mirror, order7);

        % Combined
        condRes.combined.apk7 = map_to_7(eval_apk(detComb14, testSub, 0.1), mirror, order7);
        condRes.combined.pck7 = map_to_7(eval_pck(poseComb14, testSub, 0.1), mirror, order7);
        condRes.combined.meanAPK = mean(condRes.combined.apk7);
        condRes.combined.meanPCK = mean(condRes.combined.pck7);
        condRes.combined.pckCurve = compute_pck_curve(poseComb14, testSub, thresholds, mirror, order7);

        % Refined (Ours Full)
        condRes.refined.apk7 = map_to_7(eval_apk(detRef14, testSub, 0.1), mirror, order7);
        condRes.refined.pck7 = map_to_7(eval_pck(poseRef14, testSub, 0.1), mirror, order7);
        condRes.refined.meanAPK = mean(condRes.refined.apk7);
        condRes.refined.meanPCK = mean(condRes.refined.pck7);
        condRes.refined.pckCurve = compute_pck_curve(poseRef14, testSub, thresholds, mirror, order7);

        results.degraded.(cond) = condRes;

        % Print condition results
        fprintf('  --> [%s] Baseline: APK=%.2f%%, PCK=%.2f%%\n', ...
            upper(cond), 100*condRes.baseline.meanAPK, 100*condRes.baseline.meanPCK);
        fprintf('  --> [%s] Preproc:  APK=%.2f%%, PCK=%.2f%% (Delta APK: %+.2f pp, Delta PCK: %+.2f pp)\n', ...
            upper(cond), 100*condRes.preprocess.meanAPK, 100*condRes.preprocess.meanPCK, ...
            100*(condRes.preprocess.meanAPK - condRes.baseline.meanAPK), 100*(condRes.preprocess.meanPCK - condRes.baseline.meanPCK));
        fprintf('  --> [%s] Combined: APK=%.2f%%, PCK=%.2f%% (Delta APK: %+.2f pp, Delta PCK: %+.2f pp)\n', ...
            upper(cond), 100*condRes.combined.meanAPK, 100*condRes.combined.meanPCK, ...
            100*(condRes.combined.meanAPK - condRes.baseline.meanAPK), 100*(condRes.combined.meanPCK - condRes.baseline.meanPCK));
        fprintf('  --> [%s] Refined:  APK=%.2f%%, PCK=%.2f%% (Delta APK: %+.2f pp, Delta PCK: %+.2f pp)\n', ...
            upper(cond), 100*condRes.refined.meanAPK, 100*condRes.refined.meanPCK, ...
            100*(condRes.refined.meanAPK - condRes.baseline.meanAPK), 100*(condRes.refined.meanPCK - condRes.baseline.meanPCK));
    end

    %% =============================================================
    % PART 3: GENERATE ALL PUBLICATION-QUALITY FIGURES
    % =============================================================
    fprintf('\n========================================================================\n');
    fprintf(' PART 3: GENERATING PUBLICATION-QUALITY VISUALIZATIONS & CHARTS\n');
    fprintf('========================================================================\n');

    generate_all_plots(results, conditions, jointNames7, thresholds, figuresDir);
    generate_qualitative_figures(test, model, figuresDir);

    %% =============================================================
    % PART 4: SAVE COMPLETE RESULTS
    % =============================================================
    saveFile = fullfile(codeFullDir, 'complete_evaluation_results.mat');
    save(saveFile, 'results');
    fprintf('\nComplete experimental results saved successfully to:\n  %s\n', saveFile);

    fprintf('\n========================================================================\n');
    fprintf(' EXPERIMENTAL EVALUATION SUITE COMPLETED SUCCESSFULLY!\n');
    fprintf('========================================================================\n');
end

%% -----------------------------------------------------------------
% Helper: Map 14-joint vector to 7 canonical anatomical groups
% -----------------------------------------------------------------
function vec7 = map_to_7(vec14, mirror, order7)
    vAvg = (vec14 + vec14(mirror)) / 2;
    vec7 = vAvg(order7);
end

%% -----------------------------------------------------------------
% Helper: Extract best overlapping pose struct
% -----------------------------------------------------------------
function poseObj = extract_pose(boxes, gtObj)
    poseObj = [];
    if isempty(boxes), return; end
    box = bestoverlap(boxes, gtObj, 0.3);
    if ~isempty(box)
        poseObj = box2obj(box);
    end
end

%% -----------------------------------------------------------------
% Helper: Compute PCK curve across multiple thresholds
% -----------------------------------------------------------------
function curve = compute_pck_curve(pose14, test, thresholds, mirror, order7)
    curve = zeros(1, length(thresholds));
    for t = 1:length(thresholds)
        pck14 = eval_pck(pose14, test, thresholds(t));
        pck7  = map_to_7(pck14, mirror, order7);
        curve(t) = mean(pck7);
    end
end

%% -----------------------------------------------------------------
% Helper: Print Clean Benchmark Table
% -----------------------------------------------------------------
function print_clean_summary(cleanRes, jointNames)
    fprintf('\nCLEAN PARSE TEST BENCHMARK RESULTS (205 Images):\n');
    fprintf('%-12s | %-7s %-7s %-7s %-7s %-7s %-7s %-7s | %-8s\n', ...
        'Method', jointNames{:}, 'Mean PCK');
    fprintf('------------------------------------------------------------------------------------\n');

    models = {'baseline', 'adaptiveHOG', 'combined', 'refined'};
    names  = {'Baseline', 'Adap-HOG', 'Combined', 'Refined'};

    for m = 1:length(models)
        r = cleanRes.(models{m});
        fprintf('%-12s | %6.1f  %6.1f  %6.1f  %6.1f  %6.1f  %6.1f  %6.1f  | %6.2f%%\n', ...
            names{m}, r.pck7 * 100, r.meanPCK * 100);
    end
    fprintf('------------------------------------------------------------------------------------\n');

    fprintf('%-12s | %-7s %-7s %-7s %-7s %-7s %-7s %-7s | %-8s\n', ...
        'Method', jointNames{:}, 'Mean APK');
    fprintf('------------------------------------------------------------------------------------\n');
    for m = 1:length(models)
        r = cleanRes.(models{m});
        fprintf('%-12s | %6.1f  %6.1f  %6.1f  %6.1f  %6.1f  %6.1f  %6.1f  | %6.2f%%\n', ...
            names{m}, r.apk7 * 100, r.meanAPK * 100);
    end
    fprintf('------------------------------------------------------------------------------------\n\n');
end

%% -----------------------------------------------------------------
% Helper: Generate All Charts & Curves
% -----------------------------------------------------------------
function generate_all_plots(results, conditions, jointNames7, thresholds, figuresDir)
    set(0, 'DefaultAxesFontName', 'Helvetica');
    set(0, 'DefaultTextFontName', 'Helvetica');

    % 1. Robustness Bar Chart: APK
    f1 = figure('Visible', 'off', 'Position', [100 100 900 500]);
    condLabels = {'Clean', 'Dark', 'Low Contrast', 'Noise', 'Blur'};
    methods = {'Baseline', 'Preprocess', 'Adaptive HOG', 'Combined', 'Refined'};

    apkData = zeros(5, 5);
    % Clean
    apkData(1, 1) = results.clean.baseline.meanAPK * 100;
    apkData(1, 2) = results.clean.baseline.meanAPK * 100; % approx equal
    apkData(1, 3) = results.clean.adaptiveHOG.meanAPK * 100;
    apkData(1, 4) = results.clean.combined.meanAPK * 100;
    apkData(1, 5) = results.clean.refined.meanAPK * 100;

    % Degraded conditions
    for c = 1:4
        cond = conditions{c};
        d = results.degraded.(cond);
        apkData(c+1, 1) = d.baseline.meanAPK * 100;
        apkData(c+1, 2) = d.preprocess.meanAPK * 100;
        apkData(c+1, 3) = d.adaptiveHOG.meanAPK * 100;
        apkData(c+1, 4) = d.combined.meanAPK * 100;
        apkData(c+1, 5) = d.refined.meanAPK * 100;
    end

    b = bar(apkData);
    b(1).FaceColor = [0.45 0.45 0.50]; % Baseline: Slate Gray
    b(2).FaceColor = [0.25 0.65 0.85]; % Preprocess: Sky Blue
    b(3).FaceColor = [0.85 0.55 0.20]; % Adaptive HOG: Orange
    b(4).FaceColor = [0.40 0.75 0.35]; % Combined: Green
    b(5).FaceColor = [0.85 0.20 0.30]; % Refined: Ruby Red

    set(gca, 'XTickLabel', condLabels, 'FontSize', 11, 'LineWidth', 1.2);
    ylabel('Mean APK (%)', 'FontSize', 12, 'FontWeight', 'bold');
    title('Robustness Comparison: Average Precision of Keypoints (APK@0.1)', 'FontSize', 13, 'FontWeight', 'bold');
    legend(methods, 'Location', 'northwest', 'FontSize', 10);
    grid on; box on;
    ylim([0, max(apkData(:)) * 1.15]);
    saveas(f1, fullfile(figuresDir, 'robustness_comparison_apk.png'));
    close(f1);

    % 2. Robustness Bar Chart: PCK
    f2 = figure('Visible', 'off', 'Position', [100 100 900 500]);
    pckData = zeros(5, 5);
    pckData(1, 1) = results.clean.baseline.meanPCK * 100;
    pckData(1, 2) = results.clean.baseline.meanPCK * 100;
    pckData(1, 3) = results.clean.adaptiveHOG.meanPCK * 100;
    pckData(1, 4) = results.clean.combined.meanPCK * 100;
    pckData(1, 5) = results.clean.refined.meanPCK * 100;

    for c = 1:4
        cond = conditions{c};
        d = results.degraded.(cond);
        pckData(c+1, 1) = d.baseline.meanPCK * 100;
        pckData(c+1, 2) = d.preprocess.meanPCK * 100;
        pckData(c+1, 3) = d.adaptiveHOG.meanPCK * 100;
        pckData(c+1, 4) = d.combined.meanPCK * 100;
        pckData(c+1, 5) = d.refined.meanPCK * 100;
    end

    b2 = bar(pckData);
    b2(1).FaceColor = [0.45 0.45 0.50];
    b2(2).FaceColor = [0.25 0.65 0.85];
    b2(3).FaceColor = [0.85 0.55 0.20];
    b2(4).FaceColor = [0.40 0.75 0.35];
    b2(5).FaceColor = [0.85 0.20 0.30];

    set(gca, 'XTickLabel', condLabels, 'FontSize', 11, 'LineWidth', 1.2);
    ylabel('Mean PCK (%)', 'FontSize', 12, 'FontWeight', 'bold');
    title('Robustness Comparison: Percentage of Correct Keypoints (PCK@0.1)', 'FontSize', 13, 'FontWeight', 'bold');
    legend(methods, 'Location', 'northwest', 'FontSize', 10);
    grid on; box on;
    ylim([0, max(pckData(:)) * 1.15]);
    saveas(f2, fullfile(figuresDir, 'robustness_comparison_pck.png'));
    close(f2);

    % 3. Per-Joint Breakdown (Clean Benchmark)
    f3 = figure('Visible', 'off', 'Position', [100 100 950 500]);
    jointDataClean = [
        results.clean.baseline.pck7 * 100;
        results.clean.adaptiveHOG.pck7 * 100;
        results.clean.combined.pck7 * 100;
        results.clean.refined.pck7 * 100
    ]';
    b3 = bar(jointDataClean);
    b3(1).FaceColor = [0.45 0.45 0.50];
    b3(2).FaceColor = [0.85 0.55 0.20];
    b3(3).FaceColor = [0.40 0.75 0.35];
    b3(4).FaceColor = [0.85 0.20 0.30];
    set(gca, 'XTickLabel', jointNames7, 'FontSize', 11, 'LineWidth', 1.2);
    ylabel('PCK@0.1 Accuracy (%)', 'FontSize', 12, 'FontWeight', 'bold');
    title('Clean PARSE Benchmark: Accuracy Across 7 Anatomical Keypoints', 'FontSize', 13, 'FontWeight', 'bold');
    legend({'Baseline FMP', 'Adaptive HOG', 'Combined Pipeline', 'Full Pipeline + Refinement'}, ...
           'Location', 'northeast', 'FontSize', 10);
    grid on; box on;
    ylim([0, 100]);
    saveas(f3, fullfile(figuresDir, 'keypoint_breakdown_clean.png'));
    close(f3);

    % 4. PCK vs Threshold Curve
    f4 = figure('Visible', 'off', 'Position', [100 100 850 520]);
    plot(thresholds, results.clean.baseline.pckCurve * 100, '-s', 'Color', [0.4 0.4 0.4], ...
         'LineWidth', 2.2, 'MarkerSize', 6, 'MarkerFaceColor', [0.4 0.4 0.4]); hold on;
    plot(thresholds, results.clean.adaptiveHOG.pckCurve * 100, '-^', 'Color', [0.85 0.55 0.2], ...
         'LineWidth', 2.2, 'MarkerSize', 6, 'MarkerFaceColor', [0.85 0.55 0.2]);
    plot(thresholds, results.clean.combined.pckCurve * 100, '-d', 'Color', [0.3 0.75 0.3], ...
         'LineWidth', 2.2, 'MarkerSize', 6, 'MarkerFaceColor', [0.3 0.75 0.3]);
    plot(thresholds, results.clean.refined.pckCurve * 100, '-o', 'Color', [0.85 0.2 0.3], ...
         'LineWidth', 2.5, 'MarkerSize', 7, 'MarkerFaceColor', [0.85 0.2 0.3]);
    set(gca, 'FontSize', 11, 'LineWidth', 1.2);
    xlabel('Normalized Distance Threshold (\alpha)', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Percentage of Correct Keypoints (%)', 'FontSize', 12, 'FontWeight', 'bold');
    title('PCK Curve vs. Normalized Evaluation Radius Threshold', 'FontSize', 13, 'FontWeight', 'bold');
    legend({'Baseline FMP', 'Adaptive HOG', 'Combined', 'Refined (Ours)'}, 'Location', 'southeast', 'FontSize', 10);
    grid on; box on;
    saveas(f4, fullfile(figuresDir, 'pck_threshold_curves.png'));
    close(f4);

    fprintf('  Saved charts: robustness_comparison_apk.png, robustness_comparison_pck.png, ...\n');
end

%% -----------------------------------------------------------------
% Helper: Generate Qualitative Visual Figures
% -----------------------------------------------------------------
function generate_qualitative_figures(test, model, figuresDir)
    fprintf('Generating qualitative visual comparison figures for each degradation...\n');
    conds = {'dark', 'lowcontrast', 'noise', 'blur'};
    condTitles = {'Dark Illumination', 'Low Contrast', 'Gaussian Noise', 'Gaussian Blur'};

    % Select representative test image (e.g. index 1: im0101.jpg)
    sampleIdx = 1;
    gtPoints = test(sampleIdx).obj(1).point;

    for c = 1:length(conds)
        cond = conds{c};
        imgFile = fullfile(fileparts(mfilename('fullpath')), 'data', 'PARSE_degraded', cond, 'im0101.jpg');
        imDeg = imread(imgFile);

        % Baseline
        bBase = detect_fast(imDeg, model, model.thresh);
        pBase = extract_pose(bBase, test(sampleIdx).obj);
        detBase = struct('obj', pBase);
        detBase14 = PARSE_transback(detBase);
        ptsBase = [];
        if ~isempty(detBase14.obj), ptsBase = detBase14.obj(1).point; end

        % Preprocessing + Combined
        [imPre, ~] = adaptive_preprocess(imDeg);
        bComb = adaptive_detect_fast(imPre, model, model.thresh);
        pComb = extract_pose(bComb, test(sampleIdx).obj);
        detComb = struct('obj', pComb);
        detComb14 = PARSE_transback(detComb);
        ptsComb = [];
        if ~isempty(detComb14.obj), ptsComb = detComb14.obj(1).point; end

        % Pose Refinement
        [detRef14, ~] = apply_pose_refinement(detComb14);
        ptsRef = [];
        if ~isempty(detRef14.obj), ptsRef = detRef14.obj(1).point; end

        % Render 5-panel comparison figure
        f = figure('Visible', 'off', 'Position', [50 50 1400 380]);

        subplot(1, 5, 1);
        imagesc(imDeg); axis image; axis off;
        title(sprintf('(a) %s Input', condTitles{c}), 'FontSize', 10, 'FontWeight', 'bold');

        subplot(1, 5, 2);
        plot_parse_skeleton(imDeg, ptsBase, struct('lineWidth', 3, 'markerSize', 5, 'title', '(b) Baseline FMP'));

        subplot(1, 5, 3);
        plot_parse_skeleton(imPre, ptsComb, struct('lineWidth', 3, 'markerSize', 5, 'title', '(c) Preproc + Adap-HOG'));

        subplot(1, 5, 4);
        plot_parse_skeleton(imDeg, ptsRef, struct('lineWidth', 3, 'markerSize', 5, 'title', '(d) Refined Pose (Ours)'));

        subplot(1, 5, 5);
        plot_parse_skeleton(imDeg, gtPoints, struct('lineWidth', 3, 'markerSize', 5, 'title', '(e) Ground Truth'));

        figName = sprintf('qualitative_comparison_%s.png', cond);
        saveas(f, fullfile(figuresDir, figName));
        close(f);
        fprintf('  Saved qualitative figure: %s\n', figName);
    end
end
