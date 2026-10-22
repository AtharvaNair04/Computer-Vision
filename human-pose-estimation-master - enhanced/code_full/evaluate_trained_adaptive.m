function results = evaluate_trained_adaptive(modelFile, numDegradedImages)
% EVALUATE_TRAINED_ADAPTIVE
% Compare the original pretrained FMP model against the newly trained
% adaptive-HOG FMP model.
%
% IMPORTANT: this script isolates the effect of TRAINING the model with
% adaptive HOG. It does NOT apply pose refinement. Preprocessing is also
% not mixed into this first diagnostic experiment.
%
% Run after training, for example:
%   train_adaptive_fmp(true, 1)
%   evaluate_trained_adaptive(fullfile('..','code_basic', ...
%       'PARSE_adaptive_robust_model.mat'), 25)
%
% Inputs:
%   modelFile         - .mat file containing variable 'model'
%   numDegradedImages - images per degradation condition (default 25)

    if nargin < 1 || isempty(modelFile)
        modelFile = fullfile('..','code_basic','PARSE_adaptive_robust_model.mat');
    end
    if nargin < 2 || isempty(numDegradedImages)
        numDegradedImages = 25;
    end

    clc;
    fprintf('==============================================================\n');
    fprintf(' BASELINE FMP vs TRAINED ADAPTIVE-HOG FMP\n');
    fprintf(' =============================================================\n');

    codeFullDir = fileparts(mfilename('fullpath'));
    rootDir = fileparts(codeFullDir);
    codeBasicDir = fullfile(rootDir, 'code_basic');
    oldDir = pwd;
    cleanupObj = onCleanup(@() cd(oldDir)); %#ok<NASGU>
    cd(codeFullDir);

    addpath(codeFullDir, '-begin');
    addpath(fullfile(codeFullDir, 'detection'), '-begin');
    addpath(fullfile(codeFullDir, 'evaluation'), '-begin');
    addpath(codeBasicDir, '-begin');
    addpath(fullfile(codeBasicDir, 'mex_pc'), '-begin');

    globals;

    %% Load models
    if ~isfile(modelFile)
        if ~isabsolute_local(modelFile)
            modelFile = fullfile(rootDir, 'code_basic', 'PARSE_adaptive_robust_model.mat');
        end
    end
    baseS = load(fullfile(codeBasicDir, 'PARSE_model.mat'), 'model');
    trainedS = load(modelFile, 'model', 'trainInfo');

    baseModel = baseS.model;
    trainedModel = trainedS.model;

    baseModel.thresh = min(baseModel.thresh, -1);
    trainedModel.thresh = min(trainedModel.thresh, -1);

    fprintf('Base threshold:    %.6f\n', baseModel.thresh);
    fprintf('Trained threshold: %.6f\n', trainedModel.thresh);
    fprintf('Trained from: %s\n', trainedS.trainInfo.baseModel);

    %% PARSE test set
    [~, test] = PARSE_data('PARSE');
    numTest = numel(test);

    mirror = [6 5 4 3 2 1 12 11 10 9 8 7 14 13];
    order7 = [14 9 8 7 3 2 1];
    names7 = {'Head','Shoulder','Elbow','Wrist','Hip','Knee','Ankle'};

    %% -----------------------------------------------------------
    % Clean evaluation: all 205 images
    % ------------------------------------------------------------
    fprintf('\nCLEAN PARSE (%d images)\n', numTest);

    detBase = repmat(struct('obj', []), 1, numTest);
    detTrain = repmat(struct('obj', []), 1, numTest);
    poseBase = repmat(struct('obj', []), 1, numTest);
    poseTrain = repmat(struct('obj', []), 1, numTest);

    for n = 1:numTest
        im = imread(test(n).im);

        b = detect_fast(im, baseModel, baseModel.thresh);
        poseBase(n).obj = best_pose_from_gt(b, test(n).obj);
        b = nms(b, 0.3);
        detBase(n).obj = box2obj(b);

        b = adaptive_detect_fast(im, trainedModel, trainedModel.thresh);
        poseTrain(n).obj = best_pose_from_gt(b, test(n).obj);
        b = nms(b, 0.3);
        detTrain(n).obj = box2obj(b);

        if mod(n,25)==0 || n==numTest
            fprintf('  Clean: %d/%d\n', n, numTest);
        end
    end

    clean = compute_metrics(detBase, poseBase, detTrain, poseTrain, test, mirror, order7, names7);
    print_metrics('CLEAN', clean);

    %% -----------------------------------------------------------
    % Degraded evaluation
    % ------------------------------------------------------------
    conditions = {'dark','lowcontrast','noise','blur'};
    degradedDir = fullfile(codeFullDir, 'data', 'PARSE_degraded');
    results = struct();
    results.clean = clean;

    numD = min(numDegradedImages, numTest);
    testSub = test(1:numD);

    for c = 1:numel(conditions)
        cond = conditions{c};
        fprintf('\n%s (%d images)\n', upper(cond), numD);

        detBase = repmat(struct('obj', []), 1, numD);
        detTrain = repmat(struct('obj', []), 1, numD);
        poseBase = repmat(struct('obj', []), 1, numD);
        poseTrain = repmat(struct('obj', []), 1, numD);

        for n = 1:numD
            imgNum = 100 + n;
            imgFile = fullfile(degradedDir, cond, sprintf('im%04d.jpg', imgNum));
            im = imread(imgFile);

            b = detect_fast(im, baseModel, baseModel.thresh);
            poseBase(n).obj = best_pose_from_gt(b, testSub(n).obj);
            b = nms(b, 0.3);
            detBase(n).obj = box2obj(b);

            b = adaptive_detect_fast(im, trainedModel, trainedModel.thresh);
            poseTrain(n).obj = best_pose_from_gt(b, testSub(n).obj);
            b = nms(b, 0.3);
            detTrain(n).obj = box2obj(b);
        end

        results.degraded.(cond) = compute_metrics( ...
            detBase, poseBase, detTrain, poseTrain, testSub, mirror, order7, names7);
        print_metrics(upper(cond), results.degraded.(cond));
    end

    outFile = fullfile(codeFullDir, 'trained_adaptive_evaluation_results.mat');
    save(outFile, 'results');
    fprintf('\nSaved results to:\n%s\n', outFile);
end

function tf = isabsolute_local(p)
    tf = (ispc && ~isempty(regexp(p, '^[A-Za-z]:[\\/]', 'once'))) || ...
         (~ispc && startsWith(p, filesep));
end

function poseObj = best_pose_from_gt(boxes, gtObj)
    poseObj = struct('obj', []);
    if isempty(boxes)
        return;
    end
    box = bestoverlap(boxes, gtObj, 0.3);
    if ~isempty(box)
        poseObj.obj = box2obj(box);
    end
end

function out = compute_metrics(detBase, poseBase, detTrain, poseTrain, test, mirror, order7, names7)
    apkB = eval_apk(PARSE_transback(detBase), test, 0.1);
    apkT = eval_apk(PARSE_transback(detTrain), test, 0.1);
    pckB = eval_pck(PARSE_transback(poseBase), test, 0.1);
    pckT = eval_pck(PARSE_transback(poseTrain), test, 0.1);

    apkB7 = ((apkB + apkB(mirror))/2); apkB7 = apkB7(order7);
    apkT7 = ((apkT + apkT(mirror))/2); apkT7 = apkT7(order7);
    pckB7 = ((pckB + pckB(mirror))/2); pckB7 = pckB7(order7);
    pckT7 = ((pckT + pckT(mirror))/2); pckT7 = pckT7(order7);

    out.baseline.apk7 = apkB7;
    out.trainedAdaptive.apk7 = apkT7;
    out.baseline.pck7 = pckB7;
    out.trainedAdaptive.pck7 = pckT7;
    out.baseline.meanAPK = mean(apkB7);
    out.trainedAdaptive.meanAPK = mean(apkT7);
    out.baseline.meanPCK = mean(pckB7);
    out.trainedAdaptive.meanPCK = mean(pckT7);
    out.deltaAPK = out.trainedAdaptive.meanAPK - out.baseline.meanAPK;
    out.deltaPCK = out.trainedAdaptive.meanPCK - out.baseline.meanPCK;
    out.jointNames = names7;
end

function print_metrics(label, r)
    fprintf('  Method             Mean APK     Mean PCK\n');
    fprintf('  Baseline           %8.2f%%     %8.2f%%\n', 100*r.baseline.meanAPK, 100*r.baseline.meanPCK);
    fprintf('  Trained Adaptive   %8.2f%%     %8.2f%%\n', 100*r.trainedAdaptive.meanAPK, 100*r.trainedAdaptive.meanPCK);
    fprintf('  Delta               %+7.2f pp    %+7.2f pp\n', 100*r.deltaAPK, 100*r.deltaPCK);
end
