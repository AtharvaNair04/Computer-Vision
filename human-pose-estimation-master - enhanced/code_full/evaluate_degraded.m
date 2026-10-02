function evaluate_degraded()

% EVALUATE_DEGRADED
%
% Robustness experiment for the PARSE test set.
%
% Compares:
%   1. Original FMP
%   2. FMP + Adaptive Preprocessing V2
%
% Conditions:
%   dark
%   lowcontrast
%   noise
%   blur

    clc;

    fprintf('=============================================\n');
    fprintf('  DEGRADED PARSE ROBUSTNESS EXPERIMENT\n');
    fprintf('=============================================\n\n');


    %% =========================================================
    % Paths
    % ==========================================================

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


    %% =========================================================
    % Add required paths
    % ==========================================================

    % Put code_full/detection first so the correct FMP
    % detection functions are used.

    addpath(codeFullDir, '-begin');
    addpath(detectionDir, '-begin');

    % Basic code and MEX files
    addpath(codeBasicDir);
    addpath(fullfile(codeBasicDir, 'mex_pc'));


    %% =========================================================
    % Verify important functions
    % ==========================================================

    fprintf('Using detect_fast:\n');
    disp(which('detect_fast'));

    fprintf('Using box2obj:\n');
    disp(which('box2obj'));

    fprintf('Using nms:\n');
    disp(which('nms'));

    fprintf('Using eval_apk:\n');
    disp(which('eval_apk'));

    fprintf('Using eval_pck_matched:\n');
    disp(which('eval_pck_matched'));

    fprintf('\n');


    %% =========================================================
    % Load FMP model
    % ==========================================================

    modelFile = fullfile( ...
        codeBasicDir, ...
        'PARSE_model.mat');

    fprintf('Loading FMP model...\n');

    load(modelFile, 'model');

    fprintf('Model loaded.\n\n');


    %% =========================================================
    % Use the same detection threshold as the original
    % PARSE evaluation pipeline
    % ==========================================================

    model.thresh = min(model.thresh, -1);

    fprintf('Detection threshold: %.4f\n\n', model.thresh);


    %% =========================================================
    % Load PARSE ground truth
    % ==========================================================

    fprintf('Loading PARSE ground truth...\n');

    [~, test] = PARSE_data('PARSE');

    fprintf('Test images: %d\n\n', length(test));


    %% =========================================================
    % Experimental conditions
    % ==========================================================

    conditions = { ...
        'dark', ...
        'lowcontrast', ...
        'noise', ...
        'blur'};


    %% =========================================================
    % Storage
    % ==========================================================

    results = struct();


    %% =========================================================
    % Run every degradation condition
    % ==========================================================

    for c = 1:length(conditions)

        condition = conditions{c};

        fprintf('\n');
        fprintf('=============================================\n');
        fprintf('Condition: %s\n', upper(condition));
        fprintf('=============================================\n');


        imageDir = fullfile( ...
            degradedDir, ...
            condition);


        %% -----------------------------------------------------
        % Detection containers
        % ------------------------------------------------------

        detOriginal = repmat( ...
            struct('obj', []), ...
            1, ...
            length(test));

        detEnhanced = repmat( ...
            struct('obj', []), ...
            1, ...
            length(test));


        %% =====================================================
        % Process all 205 test images
        % ======================================================

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


            im = imread(imageFile);


            %% =================================================
            % ORIGINAL FMP
            % ==================================================

            boxesOriginal = detect_fast( ...
                im, ...
                model, ...
                model.thresh);


            % Same NMS procedure as original FMP testing
            boxesOriginal = nms( ...
                boxesOriginal, ...
                0.3);


            % Convert detection boxes into object structure
            detOriginal(n).obj = ...
                box2obj(boxesOriginal);


            %% =================================================
            % ADAPTIVE PREPROCESSING V2 + FMP
            % ==================================================

            [imEnhanced, ~] = ...
                adaptive_preprocess_v2(im);


            boxesEnhanced = detect_fast( ...
                imEnhanced, ...
                model, ...
                model.thresh);


            % Same NMS procedure
            boxesEnhanced = nms( ...
                boxesEnhanced, ...
                0.3);


            % Convert into object structure
            detEnhanced(n).obj = ...
                box2obj(boxesEnhanced);


            %% =================================================
            % Progress
            % ==================================================

            if mod(n, 10) == 0

                fprintf( ...
                    '%s: %d / %d\n', ...
                    condition, ...
                    n, ...
                    length(test));

            end

        end


        %% =====================================================
        % Evaluate APK
        % ======================================================

        fprintf('\nCalculating APK...\n');


        apkOriginal = eval_apk( ...
            detOriginal, ...
            test, ...
            0.1);


        apkEnhanced = eval_apk( ...
            detEnhanced, ...
            test, ...
            0.1);


        %% =====================================================
        % Evaluate PCK
        % ======================================================

        fprintf('Calculating PCK...\n');


        pckOriginal = eval_pck_matched( ...
            detOriginal, ...
            test, ...
            0.1);


        pckEnhanced = eval_pck_matched( ...
            detEnhanced, ...
            test, ...
            0.1);


        %% =====================================================
        % Average left/right joints
        % ======================================================

        mirror = [ ...
            6 5 4 3 2 1 ...
            12 11 10 9 8 7 ...
            14 13];


        apkOriginal = ...
            (apkOriginal + apkOriginal(mirror)) / 2;


        apkEnhanced = ...
            (apkEnhanced + apkEnhanced(mirror)) / 2;


        pckOriginal = ...
            (pckOriginal + pckOriginal(mirror)) / 2;


        pckEnhanced = ...
            (pckEnhanced + pckEnhanced(mirror)) / 2;


        %% =====================================================
        % Mean scores
        % ======================================================

        meanAPKOriginal = mean(apkOriginal);

        meanAPKEnhanced = mean(apkEnhanced);

        meanPCKOriginal = mean(pckOriginal);

        meanPCKEnhanced = mean(pckEnhanced);


        %% =====================================================
        % Store results
        % ======================================================

        results.(condition).apkOriginal = ...
            apkOriginal;

        results.(condition).apkEnhanced = ...
            apkEnhanced;

        results.(condition).pckOriginal = ...
            pckOriginal;

        results.(condition).pckEnhanced = ...
            pckEnhanced;

        results.(condition).meanAPKOriginal = ...
            meanAPKOriginal;

        results.(condition).meanAPKEnhanced = ...
            meanAPKEnhanced;

        results.(condition).meanPCKOriginal = ...
            meanPCKOriginal;

        results.(condition).meanPCKEnhanced = ...
            meanPCKEnhanced;


        %% =====================================================
        % Print condition result
        % ======================================================

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


    %% =========================================================
    % FINAL SUMMARY
    % ==========================================================

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


    %% =========================================================
    % Save results
    % ==========================================================

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