% Compare original FMP with adaptive preprocessing + FMP

imageFiles = dir('images/*.jpg');

results = cell(length(imageFiles), 5);

for i = 1:length(imageFiles)

    fprintf('\nProcessing %s...\n', imageFiles(i).name);

    % Load image
    im = imread(fullfile('images', imageFiles(i).name));

    % -------------------------------------------------
    % ORIGINAL FMP
    % -------------------------------------------------
    tic;
    boxes_original = detect_fast(im, model, min(model.thresh,-1));
    boxes_original = nms(boxes_original, .1);
    time_original = toc;

    % Best detection score
    score_original = boxes_original(end);


    % -------------------------------------------------
    % ADAPTIVE PREPROCESSING
    % -------------------------------------------------
    tic;
    [enhanced, info] = adaptive_preprocess(im);

    % Enhanced FMP
    boxes_enhanced = detect_fast(enhanced, model, min(model.thresh,-1));
    boxes_enhanced = nms(boxes_enhanced, .1);
    time_enhanced = toc;

    % Best detection score
    score_enhanced = boxes_enhanced(end);


    % -------------------------------------------------
    % SCORE DIFFERENCE
    % -------------------------------------------------
    score_difference = score_enhanced - score_original;


    % -------------------------------------------------
    % STORE RESULTS
    % -------------------------------------------------
    results{i,1} = imageFiles(i).name;
    results{i,2} = score_original;
    results{i,3} = score_enhanced;
    results{i,4} = score_difference;
    results{i,5} = info.numOperations;

    fprintf('Original score : %.4f\n', score_original);
    fprintf('Enhanced score: %.4f\n', score_enhanced);
    fprintf('Difference     : %.4f\n', score_difference);
    fprintf('Operations     : %d\n', info.numOperations);
end


% -------------------------------------------------
% CREATE RESULT TABLE
% -------------------------------------------------

resultsTable = cell2table(results, ...
    'VariableNames', { ...
    'Image', ...
    'OriginalScore', ...
    'EnhancedScore', ...
    'ScoreDifference', ...
    'NumOperations'});

disp(' ');
disp('========== FINAL RESULTS ==========');
disp(resultsTable);