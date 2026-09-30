clc;

fprintf('========================================\n');
fprintf(' FMP BASELINE vs ADAPTIVE FMP\n');
fprintf(' PARSE TEST SET\n');
fprintf('========================================\n\n');

% ---------------------------------------------------------
% Paths
% ---------------------------------------------------------
addpath('../code_basic');
addpath('../code_basic/mex_pc');
addpath('../code_basic/visualization');
addpath('evaluation');

% ---------------------------------------------------------
% Load pretrained model
% ---------------------------------------------------------
load('../code_basic/PARSE_model.mat');

% ---------------------------------------------------------
% Load PARSE ground truth
% ---------------------------------------------------------
load('data/PARSE/labels.mat');

% ---------------------------------------------------------
% Create PARSE test set
% Original repository uses frames 101:305
% ---------------------------------------------------------
testims = 'data/PARSE/im%.4d.jpg';

test = struct();

for i = 101:305
    idx = i - 100;

    test(idx).im = sprintf(testims, i);
    test(idx).obj.point = ptsAll(:,:,i);
end

numImages = length(test);

fprintf('Number of test images: %d\n\n', numImages);

% ---------------------------------------------------------
% Storage for predictions
% ---------------------------------------------------------
det_original = struct('obj', cell(1,numImages));
det_enhanced = struct('obj', cell(1,numImages));

% ---------------------------------------------------------
% Process all test images
% ---------------------------------------------------------
for i = 1:numImages

    fprintf('Processing %d/%d : %s\n', ...
        i, numImages, test(i).im);

    % Load image
    im = imread(test(i).im);

    % =====================================================
    % BASELINE FMP
    % =====================================================
    boxes = detect_fast(im, model, min(model.thresh,-1));
    boxes = nms(boxes, .1);

    det_original(i).obj = box2obj(boxes);


    % =====================================================
    % ADAPTIVE PREPROCESSING + FMP
    % =====================================================
    [im_enhanced, info] = adaptive_preprocess_v2(im);

    boxes_enhanced = detect_fast( ...
        im_enhanced, model, min(model.thresh,-1));

    boxes_enhanced = nms(boxes_enhanced, .1);

    det_enhanced(i).obj = box2obj(boxes_enhanced);

end

fprintf('\nAll images processed.\n');

% ---------------------------------------------------------
% Convert 26 FMP keypoints back to 14 PARSE keypoints
% ---------------------------------------------------------
fprintf('Converting predictions...\n');

pose_original = PARSE_transback(det_original);
pose_enhanced = PARSE_transback(det_enhanced);

% ---------------------------------------------------------
% APK EVALUATION
% ---------------------------------------------------------
fprintf('\n========================================\n');
fprintf(' APK EVALUATION\n');
fprintf('========================================\n');

apk_original = eval_apk(pose_original, test, 0.1);
apk_enhanced = eval_apk(pose_enhanced, test, 0.1);

% Average left/right joints exactly as original demo
mirror = [6 5 4 3 2 1 12 11 10 9 8 7 14 13];

apk_original = ...
    (apk_original + apk_original(mirror)) / 2;

apk_enhanced = ...
    (apk_enhanced + apk_enhanced(mirror)) / 2;

% Keep the same joint ordering as PARSE_demo
order = [14 9 8 7 3 2 1];

apk_original = apk_original(order);
apk_enhanced = apk_enhanced(order);

meanapk_original = mean(apk_original);
meanapk_enhanced = mean(apk_enhanced);

fprintf('\nBaseline mean APK  : %.2f\n', ...
    meanapk_original * 100);

fprintf('Enhanced mean APK  : %.2f\n', ...
    meanapk_enhanced * 100);

fprintf('APK improvement    : %.2f\n', ...
    (meanapk_enhanced - meanapk_original) * 100);


% ---------------------------------------------------------
% PCK EVALUATION
% ---------------------------------------------------------
fprintf('\n========================================\n');
fprintf(' PCK EVALUATION\n');
fprintf('========================================\n');

pck_original = eval_pck(pose_original, test, 0.1);
pck_enhanced = eval_pck(pose_enhanced, test, 0.1);

% Average left/right joints exactly as original demo
pck_original = ...
    (pck_original + pck_original(mirror)) / 2;

pck_enhanced = ...
    (pck_enhanced + pck_enhanced(mirror)) / 2;

% Same ordering
pck_original = pck_original(order);
pck_enhanced = pck_enhanced(order);

meanpck_original = mean(pck_original);
meanpck_enhanced = mean(pck_enhanced);

fprintf('\nBaseline mean PCK  : %.2f\n', ...
    meanpck_original * 100);

fprintf('Enhanced mean PCK  : %.2f\n', ...
    meanpck_enhanced * 100);

fprintf('PCK improvement    : %.2f\n', ...
    (meanpck_enhanced - meanpck_original) * 100);


% ---------------------------------------------------------
% FINAL COMPARISON TABLE
% ---------------------------------------------------------
fprintf('\n========================================\n');
fprintf(' FINAL COMPARISON\n');
fprintf('========================================\n');

jointNames = { ...
    'Head', ...
    'Shoulder', ...
    'Elbow', ...
    'Wrist', ...
    'Hip', ...
    'Knee', ...
    'Ankle'};

fprintf('\n%-12s %-12s %-12s %-12s %-12s\n', ...
    'Joint', ...
    'APK Base', ...
    'APK Enhanced', ...
    'PCK Base', ...
    'PCK Enhanced');

fprintf('----------------------------------------------------------\n');

for j = 1:length(jointNames)

    fprintf('%-12s %10.2f %12.2f %12.2f %14.2f\n', ...
        jointNames{j}, ...
        apk_original(j)*100, ...
        apk_enhanced(j)*100, ...
        pck_original(j)*100, ...
        pck_enhanced(j)*100);

end

fprintf('\n========================================\n');
fprintf(' DONE\n');
fprintf('========================================\n');