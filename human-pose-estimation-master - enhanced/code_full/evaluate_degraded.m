function evaluate_degraded()

clc;

fprintf('=============================================\n');
fprintf('  DEGRADED PARSE ROBUSTNESS EXPERIMENT\n');
fprintf('  BASELINE vs ADAPTIVE PREPROCESSING V2\n');
fprintf('=============================================\n\n');

% =========================================================
% Paths
% =========================================================

codeFullDir = fileparts(mfilename('fullpath'));

codeBasicDir = fullfile( ...
    fileparts(codeFullDir), ...
    'code_basic');

detectionDir = fullfile( ...
    codeFullDir, ...
    'detection');

degradedDir = fullfile( ...
    codeFullDir, ...
    'data', ...
    'PARSE_degraded');

% =========================================================
% Add required paths
% =========================================================

addpath(codeFullDir, '-begin');
addpath(detectionDir, '-begin');
addpath(codeBasicDir);
addpath(fullfile(codeBasicDir, 'mex_pc'));

% =========================================================
% Verify important functions
% =========================================================

fprintf('Using detect_fast:\n');
disp(which('detect_fast'));

fprintf('Using box2obj:\n');
disp(which('box2obj'));

fprintf('Using nms:\n');
disp(which('nms'));

fprintf('Using PARSE_transback:\n');
disp(which('PARSE_transback'));

fprintf('Using eval_apk:\n');
disp(which('eval_apk'));

fprintf('Using eval_pck:\n');
disp(which('eval_pck'));

fprintf('\n');

% =========================================================
% Load FMP model
% =========================================================

modelFile = fullfile( ...
    codeBasicDir, ...
    'PARSE_model.mat');

fprintf('Loading FMP model...\n');

load(modelFile, 'model');

fprintf('Model loaded.\n\n');

% =========================================================
% Same detection threshold as clean evaluation
% =========================================================

model.thresh = min(model.thresh, -1);

fprintf('Detection threshold: %.4f\n\n', model.thresh);

% =========================================================
% Load PARSE ground truth
% =========================================================

fprintf('Loading PARSE ground truth...\n');

[~, test] = PARSE_data('PARSE');

fprintf('Test images: %d\n\n', length(test));

% =========================================================
% Experimental conditions
% =========================================================

conditions = { ...
    'dark', ...
    'lowcontrast', ...
    'noise', ...
    'blur'};

% =========================================================
% Storage
% =========================================================

results = struct();

% =========================================================
% Run every degradation condition
% =========================================================

for c = 1:length(conditions)

    condition = conditions{c};

    fprintf('\n');
    fprintf('=============================================\n');
    fprintf('Condition: %s\n', upper(condition));
    fprintf('=============================================\n');

    imageDir = fullfile( ...
        degradedDir, ...
        condition);

    % -----------------------------------------------------
    % Detection containers
    % -----------------------------------------------------

    detOriginal = repmat( ...
        struct('obj', []), ...
        1, ...
        length(test));

    detEnhanced = repmat( ...
        struct('obj', []), ...
        1, ...
        length(test));

    % =====================================================
    % Preprocessing statistics
    % =====================================================

    countGamma = 0;
    countBrightness = 0;
    countContrast = 0;
    countNoise = 0;
    countSharpness = 0;
    countNoChange = 0;

    % =====================================================
    % Process all test images
    % =====================================================

    for n = 1:length(test)

        imageNumber = 100 + n;

        filename = sprintf( ...
            'im%04d.jpg', ...
            imageNumber);

        imageFile = fullfile( ...
            imageDir, ...
            filename);

        if ~exist(imageFile, 'file')

            error( ...
                'Missing degraded image: %s', ...
                imageFile);

        end

        % -------------------------------------------------
        % Load degraded image
        % -------------------------------------------------

        im = imread(imageFile);

        % =================================================
        % ORIGINAL FMP
        % =================================================

        boxesOriginal = detect_fast( ...
            im, ...
            model, ...
            model.thresh);

        boxesOriginal = nms( ...
            boxesOriginal, ...
            0.1);

        detOriginal(n).obj = ...
            box2obj(boxesOriginal);

        % =================================================
        % ADAPTIVE PREPROCESSING V2 + FMP
        % =================================================

        [imEnhanced, info] = ...
            adaptive_preprocess_v2(im);

        % -------------------------------------------------
        % Count preprocessing operations
        % -------------------------------------------------

        if info.numOperations == 0

            countNoChange = countNoChange + 1;

        end

        for k = 1:length(info.operations)

            op = lower(info.operations{k});

            if contains(op, 'gamma')

                countGamma = countGamma + 1;

            elseif contains(op, 'brightness')

                countBrightness = countBrightness + 1;

            elseif contains(op, 'contrast')

                countContrast = countContrast + 1;

            elseif contains(op, 'noise')

                countNoise = countNoise + 1;

            elseif contains(op, 'sharpness')

                countSharpness = countSharpness + 1;

            end

        end

        % -------------------------------------------------
        % FMP on enhanced image
        % -------------------------------------------------

        boxesEnhanced = detect_fast( ...
            imEnhanced, ...
            model, ...
            model.thresh);

        boxesEnhanced = nms( ...
            boxesEnhanced, ...
            0.1);

        detEnhanced(n).obj = ...
            box2obj(boxesEnhanced);

        % -------------------------------------------------
        % Progress
        % -------------------------------------------------

        if mod(n, 10) == 0

            fprintf( ...
                '%s: %d / %d\n', ...
                condition, ...
                n, ...
                length(test));

        end

    end

    % =====================================================
    % Print preprocessing statistics
    % =====================================================

    fprintf('\n');
    fprintf('---------------------------------------------\n');
    fprintf('PREPROCESSING ACTIVITY: %s\n', upper(condition));
    fprintf('---------------------------------------------\n');

    fprintf( ...
        'Gamma correction      : %d / %d\n', ...
        countGamma, length(test));

    fprintf( ...
        'Brightness correction : %d / %d\n', ...
        countBrightness, length(test));

    fprintf( ...
        'Contrast enhancement  : %d / %d\n', ...
        countContrast, length(test));

    fprintf( ...
        'High noise detected   : %d / %d\n', ...
        countNoise, length(test));

    fprintf( ...
        'Low sharpness detected: %d / %d\n', ...
        countSharpness, length(test));

    fprintf( ...
        'No change             : %d / %d\n', ...
        countNoChange, length(test));

    % =====================================================
    % Convert FMP 26 keypoints to PARSE 14 keypoints
    % =====================================================

    fprintf('\nConverting predictions...\n');

    poseOriginal = PARSE_transback(detOriginal);

    poseEnhanced = PARSE_transback(detEnhanced);

    % =====================================================
    % APK EVALUATION
    % =====================================================

    fprintf('\nCalculating APK...\n');

    apkOriginal = eval_apk( ...
        poseOriginal, ...
        test, ...
        0.1);

    apkEnhanced = eval_apk( ...
        poseEnhanced, ...
        test, ...
        0.1);

    % -----------------------------------------------------
    % Average left/right joints
    % -----------------------------------------------------

    mirror = [ ...
        6 5 4 3 2 1 ...
        12 11 10 9 8 7 ...
        14 13];

    apkOriginal = ...
        (apkOriginal + apkOriginal(mirror)) / 2;

    apkEnhanced = ...
        (apkEnhanced + apkEnhanced(mirror)) / 2;

    % -----------------------------------------------------
    % Same PARSE joint ordering
    % -----------------------------------------------------

    order = [14 9 8 7 3 2 1];

    apkOriginal = apkOriginal(order);

    apkEnhanced = apkEnhanced(order);

    % -----------------------------------------------------
    % Mean APK
    % -----------------------------------------------------

    meanAPKOriginal = mean(apkOriginal);

    meanAPKEnhanced = mean(apkEnhanced);

    % =====================================================
    % PCK EVALUATION
    % =====================================================

    fprintf('Calculating PCK...\n');

    pckOriginal = eval_pck( ...
        poseOriginal, ...
        test, ...
        0.1);

    pckEnhanced = eval_pck( ...
        poseEnhanced, ...
        test, ...
        0.1);

    % -----------------------------------------------------
    % Average left/right joints
    % -----------------------------------------------------

    pckOriginal = ...
        (pckOriginal + pckOriginal(mirror)) / 2;

    pckEnhanced = ...
        (pckEnhanced + pckEnhanced(mirror)) / 2;

    % -----------------------------------------------------
    % Same PARSE joint ordering
    % -----------------------------------------------------

    pckOriginal = pckOriginal(order);

    pckEnhanced = pckEnhanced(order);

    % -----------------------------------------------------
    % Mean PCK
    % -----------------------------------------------------

    meanPCKOriginal = mean(pckOriginal);

    meanPCKEnhanced = mean(pckEnhanced);

    % =====================================================
    % Store results
    % =====================================================

    results.(condition).apkOriginal = apkOriginal;
    results.(condition).apkEnhanced = apkEnhanced;

    results.(condition).pckOriginal = pckOriginal;
    results.(condition).pckEnhanced = pckEnhanced;

    results.(condition).meanAPKOriginal = ...
        meanAPKOriginal;

    results.(condition).meanAPKEnhanced = ...
        meanAPKEnhanced;

    results.(condition).meanPCKOriginal = ...
        meanPCKOriginal;

    results.(condition).meanPCKEnhanced = ...
        meanPCKEnhanced;

    % -----------------------------------------------------
    % Store preprocessing statistics
    % -----------------------------------------------------

    results.(condition).countGamma = countGamma;
    results.(condition).countBrightness = countBrightness;
    results.(condition).countContrast = countContrast;
    results.(condition).countNoise = countNoise;
    results.(condition).countSharpness = countSharpness;
    results.(condition).countNoChange = countNoChange;

    % =====================================================
    % Print condition result
    % =====================================================

    fprintf('\n');
    fprintf('RESULT: %s\n', upper(condition));
    fprintf('---------------------------------------------\n');

    fprintf( ...
        'Original APK : %.2f\n', ...
        100 * meanAPKOriginal);

    fprintf( ...
        'Enhanced APK : %.2f\n', ...
        100 * meanAPKEnhanced);

    fprintf( ...
        'APK change   : %.2f pp\n', ...
        100 * (meanAPKEnhanced - meanAPKOriginal));

    fprintf('\n');

    fprintf( ...
        'Original PCK : %.2f\n', ...
        100 * meanPCKOriginal);

    fprintf( ...
        'Enhanced PCK : %.2f\n', ...
        100 * meanPCKEnhanced);

    fprintf( ...
        'PCK change   : %.2f pp\n', ...
        100 * (meanPCKEnhanced - meanPCKOriginal));

end

% =========================================================
% FINAL SUMMARY
% =========================================================

fprintf('\n\n');

fprintf( ...
    '============================================================\n');

fprintf( ...
    '              FINAL ROBUSTNESS RESULTS\n');

fprintf( ...
    '============================================================\n');

fprintf('\n');

fprintf( ...
    '%-15s %-12s %-12s %-12s %-12s %-12s %-12s\n', ...
    'Condition', ...
    'Orig APK', ...
    'Enh APK', ...
    'Delta APK', ...
    'Orig PCK', ...
    'Enh PCK', ...
    'Delta PCK');

fprintf( ...
    '------------------------------------------------------------\n');

for c = 1:length(conditions)

    condition = conditions{c};

    r = results.(condition);

    fprintf( ...
        '%-15s %10.2f %12.2f %12.2f %12.2f %12.2f %12.2f\n', ...
        condition, ...
        100 * r.meanAPKOriginal, ...
        100 * r.meanAPKEnhanced, ...
        100 * r.meanAPKEnhanced - ...
            100 * r.meanAPKOriginal, ...
        100 * r.meanPCKOriginal, ...
        100 * r.meanPCKEnhanced, ...
        100 * r.meanPCKEnhanced - ...
            100 * r.meanPCKOriginal);

end

fprintf( ...
    '============================================================\n');

% =========================================================
% Save results
% =========================================================

outputFile = fullfile( ...
    codeFullDir, ...
    'degraded_results.mat');

save( ...
    outputFile, ...
    'results');

fprintf('\n');

fprintf( ...
    'Results saved to:\n%s\n', ...
    outputFile);

fprintf('\n');

fprintf('Experiment completed successfully.\n');

end