function results = evaluate_adaptive_features()

clc;

%% ================================================================
% PROJECT PATHS
% ================================================================

root = 'C:\Users\athar\OneDrive\Desktop\Computer-Vision\human-pose-estimation-master - enhanced';

basicDir = fullfile(root,'code_basic');
fullDir  = fullfile(root,'code_full');

% PARSE_data.m expects code_full as the current directory
cd(fullDir);

% Make enhanced functions available
addpath(basicDir);

%% ================================================================
% LOAD MODEL
% ================================================================

load(fullfile(basicDir,'PARSE_model.mat'),'model');

fprintf('\n============================================\n');
fprintf(' FMP BASELINE vs ADAPTIVE FEATURE EXTRACTION\n');
fprintf('============================================\n');

fprintf('Model threshold: %.4f\n', model.thresh);

%% ================================================================
% LOAD PARSE TEST SET
% ================================================================

[~, test] = PARSE_data('PARSE');

N = length(test);

fprintf('PARSE test images: %d\n', N);

%% ================================================================
% VERIFY IMPORTANT FUNCTIONS
% ================================================================

fprintf('\nFunction paths:\n');

fprintf('testmodel:\n');
which testmodel

fprintf('adaptive_testmodel:\n');
which adaptive_testmodel

fprintf('detect_fast:\n');
which detect_fast

fprintf('adaptive_detect_fast:\n');
which adaptive_detect_fast

fprintf('bestoverlap:\n');
which bestoverlap

fprintf('nms:\n');
which nms

fprintf('PARSE_transback:\n');
which PARSE_transback

fprintf('eval_apk:\n');
which eval_apk

fprintf('eval_pck_matched:\n');
which eval_pck_matched

%% ================================================================
% BASELINE FMP
% ================================================================

fprintf('\n============================================\n');
fprintf(' RUNNING BASELINE FMP\n');
fprintf('============================================\n');

% Separate cache name.
% This does NOT overwrite the original repository cache.
baselineName = 'PARSE_baseline_eval';

[detBase, poseBase] = testmodel( ...
    baselineName, ...
    model, ...
    test, ...
    '_adaptive_comparison');

%% ================================================================
% ADAPTIVE FEATURE FMP
% ================================================================

fprintf('\n============================================\n');
fprintf(' RUNNING ADAPTIVE FEATURE FMP\n');
fprintf('============================================\n');

adaptiveName = 'PARSE_adaptive_eval';

[detAdaptive, poseAdaptive] = adaptive_testmodel( ...
    adaptiveName, ...
    model, ...
    test, ...
    '_adaptive_comparison');

%% ================================================================
% KEEP RAW 26-POINT DETECTIONS
% ================================================================
%
% box2obj() produces the FMP 26-point representation.
%
% We keep these raw detections because:
%
%   PCK evaluator already performs the 26 -> 14 conversion.
%
% For APK, we create separate PARSE-format copies below.

detBaseRaw = detBase;
detAdaptiveRaw = detAdaptive;

%% ================================================================
% CONVERT FMP 26-POINT OUTPUT TO PARSE 14-POINT FORMAT
% ================================================================
%
% PARSE_transback.m is part of the original project and maps:
%
%       26 FMP keypoints
%              ->
%       14 PARSE keypoints
%
% This conversion is required before eval_apk().

fprintf('\n============================================\n');
fprintf(' CONVERTING FMP OUTPUT TO PARSE FORMAT\n');
fprintf('============================================\n');

detBasePARSE = PARSE_transback(detBaseRaw);
detAdaptivePARSE = PARSE_transback(detAdaptiveRaw);

fprintf('Baseline detections converted.\n');
fprintf('Adaptive detections converted.\n');

%% ================================================================
% APK EVALUATION
% ================================================================

fprintf('\n============================================\n');
fprintf(' APK EVALUATION\n');
fprintf('============================================\n');

% IMPORTANT:
% eval_apk expects detections in the same 14-point representation
% as the PARSE ground truth.

[apkBase, precBase, recBase] = eval_apk( ...
    detBasePARSE, ...
    test);

[apkAdaptive, precAdaptive, recAdaptive] = eval_apk( ...
    detAdaptivePARSE, ...
    test);

meanAPKBase = mean(apkBase);
meanAPKAdaptive = mean(apkAdaptive);

%% ================================================================
% PCK EVALUATION
% ================================================================
%
% IMPORTANT:
% eval_pck_matched.m already contains the 26 -> 14 Transback
% conversion internally.
%
% Therefore we MUST pass the RAW 26-point detections here.
%
% Do NOT pass detBasePARSE or detAdaptivePARSE.

fprintf('\n============================================\n');
fprintf(' PCK EVALUATION\n');
fprintf('============================================\n');

pckBase = eval_pck_matched( ...
    detBaseRaw, ...
    test, ...
    0.5);

pckAdaptive = eval_pck_matched( ...
    detAdaptiveRaw, ...
    test, ...
    0.5);

meanPCKBase = mean(pckBase);
meanPCKAdaptive = mean(pckAdaptive);

%% ================================================================
% JOINT NAMES
% ================================================================
%
% For now use neutral names because we have not yet verified the
% exact anatomical ordering of all 14 PARSE joints.
%
% We can map these to anatomical names later after verifying
% labels.mat / PARSE annotation ordering.

jointNames = arrayfun( ...
    @(x) sprintf('Joint %d', x), ...
    1:14, ...
    'UniformOutput', false);

%% ================================================================
% DISPLAY APK RESULTS
% ================================================================

fprintf('\n============================================\n');
fprintf(' APK RESULTS\n');
fprintf('============================================\n');

fprintf('\n');
fprintf('%-15s %12s %12s %12s\n', ...
    'Joint', ...
    'Baseline', ...
    'Adaptive', ...
    'Change');

fprintf('------------------------------------------------------------\n');

for j = 1:length(apkBase)

    delta = (apkAdaptive(j) - apkBase(j)) * 100;

    fprintf('%-15s %11.2f%% %11.2f%% %+11.2f pp\n', ...
        jointNames{j}, ...
        apkBase(j) * 100, ...
        apkAdaptive(j) * 100, ...
        delta);

end

fprintf('------------------------------------------------------------\n');

fprintf('%-15s %11.2f%% %11.2f%% %+11.2f pp\n', ...
    'MEAN APK', ...
    meanAPKBase * 100, ...
    meanAPKAdaptive * 100, ...
    (meanAPKAdaptive - meanAPKBase) * 100);

%% ================================================================
% DISPLAY PCK RESULTS
% ================================================================

fprintf('\n============================================\n');
fprintf(' PCK RESULTS\n');
fprintf('============================================\n');

fprintf('\n');
fprintf('%-15s %12s %12s %12s\n', ...
    'Joint', ...
    'Baseline', ...
    'Adaptive', ...
    'Change');

fprintf('------------------------------------------------------------\n');

for j = 1:length(pckBase)

    delta = (pckAdaptive(j) - pckBase(j)) * 100;

    fprintf('%-15s %11.2f%% %11.2f%% %+11.2f pp\n', ...
        jointNames{j}, ...
        pckBase(j) * 100, ...
        pckAdaptive(j) * 100, ...
        delta);

end

fprintf('------------------------------------------------------------\n');

fprintf('%-15s %11.2f%% %11.2f%% %+11.2f pp\n', ...
    'MEAN PCK', ...
    meanPCKBase * 100, ...
    meanPCKAdaptive * 100, ...
    (meanPCKAdaptive - meanPCKBase) * 100);

%% ================================================================
% STORE RESULTS
% ================================================================

results = struct();

%% Baseline

results.baseline.apk = apkBase;
results.baseline.pck = pckBase;

results.baseline.meanAPK = meanAPKBase;
results.baseline.meanPCK = meanPCKBase;

results.baseline.prec = precBase;
results.baseline.rec = recBase;

%% Adaptive

results.adaptive.apk = apkAdaptive;
results.adaptive.pck = pckAdaptive;

results.adaptive.meanAPK = meanAPKAdaptive;
results.adaptive.meanPCK = meanPCKAdaptive;

results.adaptive.prec = precAdaptive;
results.adaptive.rec = recAdaptive;

%% Difference

results.deltaAPK = apkAdaptive - apkBase;
results.deltaPCK = pckAdaptive - pckBase;

results.meanDeltaAPK = meanAPKAdaptive - meanAPKBase;
results.meanDeltaPCK = meanPCKAdaptive - meanPCKBase;

%% ================================================================
% SAVE RESULTS
% ================================================================

resultFile = fullfile( ...
    basicDir, ...
    'adaptive_feature_comparison.mat');

save( ...
    resultFile, ...
    'results', ...
    'detBaseRaw', ...
    'detAdaptiveRaw', ...
    'poseBase', ...
    'poseAdaptive');

%% ================================================================
% FINAL SUMMARY
% ================================================================

fprintf('\n============================================\n');
fprintf(' FINAL SUMMARY\n');
fprintf('============================================\n');

fprintf('\nPARSE test images : %d\n', N);

fprintf('\nMean APK:\n');
fprintf('  Baseline : %.2f%%\n', meanAPKBase * 100);
fprintf('  Adaptive : %.2f%%\n', meanAPKAdaptive * 100);
fprintf('  Change   : %+0.2f percentage points\n', ...
    (meanAPKAdaptive - meanAPKBase) * 100);

fprintf('\nMean PCK:\n');
fprintf('  Baseline : %.2f%%\n', meanPCKBase * 100);
fprintf('  Adaptive : %.2f%%\n', meanPCKAdaptive * 100);
fprintf('  Change   : %+0.2f percentage points\n', ...
    (meanPCKAdaptive - meanPCKBase) * 100);

fprintf('\n============================================\n');
fprintf(' RESULTS SAVED\n');
fprintf('============================================\n');

fprintf('%s\n', resultFile);

fprintf('\nEvaluation completed successfully.\n');

end