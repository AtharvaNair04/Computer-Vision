function results = run_buffy_experiments(numTestImages)
% RUN_BUFFY_EXPERIMENTS
% Evaluates Baseline FMP vs. Enhanced Pipeline on BUFFY dataset
% Protocol: 10 Upper-Body Keypoints (Head, Shoulder, Elbow, Wrist, Hip)

    if nargin < 1 || isempty(numTestImages)
        numTestImages = 25;
    end

    clc;
    fprintf('========================================================================\n');
    fprintf('  BUFFY DATASET BENCHMARK: BASELINE vs. ENHANCED FMP PIPELINE\n');
    fprintf('========================================================================\n\n');

    globals;
    addpath(fullfile('..', 'code_basic'));
    addpath(fullfile('..', 'code_basic', 'mex_pc'));
    addpath(fullfile('mex_pc'));

    % Load BUFFY Model
    modelFile = fullfile('..', 'code_basic', 'BUFFY_model.mat');
    fprintf('Loading BUFFY model from %s...\n', modelFile);
    load(modelFile, 'model');
    model.thresh = min(model.thresh, -1);

    % Load BUFFY Test Data
    [~, test] = BUFFY_data('BUFFY');
    numTest = min(numTestImages, length(test));
    testSub = test(1:numTest);
    fprintf('Evaluating on %d BUFFY test images...\n\n', numTest);

    mirror = [2 1 5 6 3 4 8 7 10 9];
    order5 = [1 3 4 7 9]; % Head, Shoulder, Elbow, Wrist, Hip
    jointNames5 = {'Head', 'Shoulder', 'Elbow', 'Wrist', 'Hip'};

    detBaseRaw = repmat(struct('obj', []), 1, numTest);
    detEnhRaw  = repmat(struct('obj', []), 1, numTest);
    poseBaseRaw = repmat(struct('obj', []), 1, numTest);
    poseEnhRaw  = repmat(struct('obj', []), 1, numTest);

    ticB = tic;
    for n = 1:numTest
        im = imread(testSub(n).im);
        gtObj = testSub(n).obj;

        % 1. Baseline FMP
        bBase = detect_fast(im, model, model.thresh);
        if ~isempty(bBase)
            bestBox = bestoverlap(bBase, gtObj, 0.2);
            if ~isempty(bestBox), poseBaseRaw(n).obj = box2obj(bestBox); end
            bBase = nms(bBase, 0.2);
            detBaseRaw(n).obj = box2obj(bBase);
        end

        % 2. Enhanced (Adaptive Preprocessing + FMP)
        [imEnh, ~] = adaptive_preprocess(im);
        bEnh = detect_fast(imEnh, model, model.thresh);
        if ~isempty(bEnh)
            bestBoxE = bestoverlap(bEnh, gtObj, 0.2);
            if ~isempty(bestBoxE), poseEnhRaw(n).obj = box2obj(bestBoxE); end
            bEnh = nms(bEnh, 0.2);
            detEnhRaw(n).obj = box2obj(bEnh);
        end

        if mod(n, 5) == 0 || n == numTest
            fprintf('  BUFFY Processed %d/%d images (elapsed: %.1fs)\n', n, numTest, toc(ticB));
        end
    end

    % Map to 10 BUFFY joints
    detBase10  = BUFFY_transback(detBaseRaw);
    poseBase10 = BUFFY_transback(poseBaseRaw);
    detEnh10   = BUFFY_transback(detEnhRaw);
    poseEnh10  = BUFFY_transback(poseEnhRaw);

    % Compute Metrics
    % Baseline
    apkBase = eval_apk(detBase10, testSub, 0.2);
    apkBase5 = (apkBase + apkBase(mirror)) / 2;
    apkBase5 = apkBase5(order5);

    pckBase = eval_pck(poseBase10, testSub, 0.2);
    pckBase5 = (pckBase + pckBase(mirror)) / 2;
    pckBase5 = pckBase5(order5);

    % Enhanced
    apkEnh = eval_apk(detEnh10, testSub, 0.2);
    apkEnh5 = (apkEnh + apkEnh(mirror)) / 2;
    apkEnh5 = apkEnh5(order5);

    pckEnh = eval_pck(poseEnh10, testSub, 0.2);
    pckEnh5 = (pckEnh + pckEnh(mirror)) / 2;
    pckEnh5 = pckEnh5(order5);

    % Print Comparison Table
    fprintf('\n========================================================================\n');
    fprintf('  BUFFY DATASET BENCHMARK RESULTS (%d Images)\n', numTest);
    fprintf('========================================================================\n');
    fprintf('%-12s | %-8s %-8s %-8s %-8s %-8s | %-9s\n', ...
        'Metric / Joint', jointNames5{:}, 'Average');
    fprintf('------------------------------------------------------------------------\n');
    fprintf('%-12s | %6.1f%%  %6.1f%%  %6.1f%%  %6.1f%%  %6.1f%% |  %6.2f%%\n', ...
        'PCK Baseline', pckBase5 * 100, mean(pckBase5) * 100);
    fprintf('%-12s | %6.1f%%  %6.1f%%  %6.1f%%  %6.1f%%  %6.1f%% |  %6.2f%%\n', ...
        'PCK Enhanced', pckEnh5 * 100, mean(pckEnh5) * 100);
    fprintf('%-12s | %+6.1fpp %+6.1fpp %+6.1fpp %+6.1fpp %+6.1fpp | %+6.2fpp\n', ...
        'Delta PCK', (pckEnh5 - pckBase5) * 100, (mean(pckEnh5) - mean(pckBase5)) * 100);
    fprintf('------------------------------------------------------------------------\n');
    fprintf('%-12s | %6.1f%%  %6.1f%%  %6.1f%%  %6.1f%%  %6.1f%% |  %6.2f%%\n', ...
        'APK Baseline', apkBase5 * 100, mean(apkBase5) * 100);
    fprintf('%-12s | %6.1f%%  %6.1f%%  %6.1f%%  %6.1f%%  %6.1f%% |  %6.2f%%\n', ...
        'APK Enhanced', apkEnh5 * 100, mean(apkEnh5) * 100);
    fprintf('%-12s | %+6.1fpp %+6.1fpp %+6.1fpp %+6.1fpp %+6.1fpp | %+6.2fpp\n', ...
        'Delta APK', (apkEnh5 - apkBase5) * 100, (mean(apkEnh5) - mean(apkBase5)) * 100);
    fprintf('========================================================================\n\n');

    results.pckBase5 = pckBase5;
    results.pckEnh5  = pckEnh5;
    results.apkBase5 = apkBase5;
    results.apkEnh5  = apkEnh5;
    results.meanPCK_base = mean(pckBase5);
    results.meanPCK_enh  = mean(pckEnh5);
    results.meanAPK_base = mean(apkBase5);
    results.meanAPK_enh  = mean(apkEnh5);

    save(fullfile('cache', sprintf('BUFFY_eval_N%d.mat', numTest)), 'results');
end
