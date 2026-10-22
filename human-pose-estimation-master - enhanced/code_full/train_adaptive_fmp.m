function model = train_adaptive_fmp(useDegradedAugmentation, numIterations)
% TRAIN_ADAPTIVE_FMP
% Fine-tune the pretrained PARSE FMP model using the SAME adaptive HOG
% feature extractor that is used during enhanced inference.
%
% This fixes the current experimental mismatch:
%   old setup = adaptive HOG at test time + original-HOG-trained model
%   new setup = adaptive HOG at training time + adaptive-HOG-trained model
%
% Inputs:
%   useDegradedAugmentation - true (recommended) adds dark/contrast/noise/
%                             blur PARSE training examples.
%   numIterations           - structured-SVM fine-tuning iterations.
%                             Start with 1 for the first experiment.
%
% Output:
%   model - fine-tuned adaptive FMP model
%
% The original PARSE_model.mat is NEVER overwritten.

    if nargin < 1 || isempty(useDegradedAugmentation)
        useDegradedAugmentation = true;
    end
    if nargin < 2 || isempty(numIterations)
        numIterations = 1;
    end

    clc;
    fprintf('==============================================================\n');
    fprintf(' ADAPTIVE FMP FINE-TUNING\n');
    fprintf('==============================================================\n');

    codeFullDir = fileparts(mfilename('fullpath'));
    rootDir = fileparts(codeFullDir);
    oldDir = pwd;
    cleanupObj = onCleanup(@() cd(oldDir)); %#ok<NASGU>
    cd(codeFullDir);
    codeBasicDir = fullfile(rootDir, 'code_basic');

    % Put adaptive learning code first, then required original learning,
    % detection, evaluation and MEX paths.
    addpath(codeFullDir, '-begin');
    addpath(fullfile(codeFullDir, 'learning'), '-begin');
    addpath(fullfile(codeFullDir, 'detection'), '-begin');
    addpath(codeBasicDir, '-begin');
    addpath(fullfile(codeBasicDir, 'mex_pc'), '-begin');
    addpath(fullfile(codeFullDir, 'mex_pc'), '-begin');

    globals;

    %% -----------------------------------------------------------
    % Verify adaptive feature extractor
    % ------------------------------------------------------------
    fprintf('\nAdaptive feature extractor:\n');
    disp(which('adaptive_features'));
    fprintf('Adaptive feature pyramid:\n');
    disp(which('adaptive_featpyramid'));
    fprintf('Adaptive training detector:\n');
    disp(which('adaptive_learning_detect'));

    if isempty(which('adaptive_features'))
        error(['adaptive_features MEX is missing. Run the Windows ', ...
               'compile step for code_basic first.']);
    end

    %% -----------------------------------------------------------
    % Load the pretrained FMP model
    % ------------------------------------------------------------
    baseModelFile = fullfile(codeBasicDir, 'PARSE_model.mat');
    S = load(baseModelFile, 'model');
    model = S.model;

    fprintf('\nBase model loaded from:\n%s\n', baseModelFile);
    fprintf('  sbin     = %d\n', model.sbin);
    fprintf('  interval = %d\n', model.interval);
    fprintf('  filters  = %d\n', numel(model.filters));
    fprintf('  len      = %d\n', model.len);
    fprintf('  threshold= %.6f\n', model.thresh);

    %% -----------------------------------------------------------
    % Prepare PARSE positives
    % ------------------------------------------------------------
    if useDegradedAugmentation
        pos = make_adaptive_training_set();
    else
        [pos, ~] = PARSE_data('PARSE');
        pos = pos(1:100); % original clean training frames only
    end

    % Convert 26 keypoints into the FMP part boxes exactly as PARSE_demo.m
    PARA.parent = [0 1 2 3 4 5 6 3 8 9 10 11 12 13 2 15 16 17 18 15 20 21 22 23 24 25];
    pos = point2box(pos, PARA.parent);

    fprintf('\nPositive examples used for fine-tuning: %d\n', numel(pos));

    %% -----------------------------------------------------------
    % Load INRIA negatives
    % ------------------------------------------------------------
    fprintf('\nLoading INRIA negatives...\n');
    neg = INRIA_data('INRIA');

    % Fail early if the full negative dataset is not present.
    missing = false;
    for n = 1:numel(neg)
        if ~exist(neg(n).im, 'file')
            missing = true;
            break;
        end
    end
    if missing
        error(['Full INRIA negative training data is required for ', ...
               'structured fine-tuning. Missing file example: ', neg(n).im]);
    end
    fprintf('INRIA negatives available: %d\n', numel(neg));

    %% -----------------------------------------------------------
    % Fine-tune from the pretrained model
    % ------------------------------------------------------------
    % warp=0 -> latent positive search. This is important because the
    % adaptive learning detector must generate its positive feature
    % vectors through adaptive_featpyramid().
    %
    % C and positive weighting are the original defaults used by the
    % repository: C=0.002 and wpos=2.
    trainName = 'PARSE_adaptive_finetune';

    fprintf('\nStarting adaptive structured-SVM fine-tuning...\n');
    fprintf('Iterations: %d\n', numIterations);
    fprintf('This can take substantial time and memory.\n\n');

    tic;
    model = adaptive_train( ...
        trainName, ...
        model, ...
        pos, ...
        neg, ...
        0, ...              % warp = 0: latent positives
        numIterations);     % iterations
    elapsedMin = toc / 60;

    %% -----------------------------------------------------------
    % Save under a new name
    % ------------------------------------------------------------
    if useDegradedAugmentation
        outFile = fullfile(codeBasicDir, 'PARSE_adaptive_robust_model.mat');
    else
        outFile = fullfile(codeBasicDir, 'PARSE_adaptive_clean_model.mat');
    end

    trainInfo = struct();
    trainInfo.useDegradedAugmentation = useDegradedAugmentation;
    trainInfo.numIterations = numIterations;
    trainInfo.numPositives = numel(pos);
    trainInfo.trainingMinutes = elapsedMin;
    trainInfo.baseModel = baseModelFile;
    trainInfo.featureExtractor = which('adaptive_features');

    save(outFile, 'model', 'trainInfo', '-v7');

    fprintf('\n==============================================================\n');
    fprintf(' ADAPTIVE FMP TRAINING COMPLETE\n');
    fprintf(' Time: %.2f minutes\n', elapsedMin);
    fprintf(' New model threshold: %.6f\n', model.thresh);
    fprintf(' Saved to:\n%s\n', outFile);
    fprintf('==============================================================\n');
end
