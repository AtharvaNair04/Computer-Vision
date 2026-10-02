function results = evaluate_baseline_same_eval()

clc;

root = 'C:\Users\athar\OneDrive\Desktop\Computer-Vision\human-pose-estimation-master - enhanced';

basicDir = fullfile(root,'code_basic');
fullDir  = fullfile(root,'code_full');

cd(fullDir);
addpath(basicDir);

load(fullfile(basicDir,'PARSE_model.mat'),'model');

thresh = min(model.thresh,-1);

[~,test] = PARSE_data('PARSE');

N = length(test);

fprintf('\n============================================\n');
fprintf(' BASELINE SAME-EVALUATION BENCHMARK\n');
fprintf('============================================\n');
fprintf('Test images: %d\n',N);

det = repmat(struct('obj',[]),1,N);

for n = 1:N

    fprintf('Processing image %d / %d\n',n,N);

    im = imread(test(n).im);

    boxes = detect_fast(im,model,thresh);

    if isempty(boxes)
        det(n).obj = [];
    else
        det(n).obj = box2obj(boxes);
    end

end

fprintf('\nCalculating APK...\n');

[apk,prec,rec] = eval_apk(det,test);

fprintf('Calculating PCK...\n');

pck = eval_pck_matched(det,test,0.5);

results.apk = apk;
results.pck = pck;

results.meanAPK = mean(apk);
results.meanPCK = mean(pck);

fprintf('\n============================================\n');
fprintf(' BASELINE SAME-EVALUATION RESULTS\n');
fprintf('============================================\n');

fprintf('Mean APK : %.2f%%\n',results.meanAPK * 100);
fprintf('Mean PCK : %.2f%%\n',results.meanPCK * 100);

fprintf('\nJoint-wise APK:\n');
disp(apk * 100);

fprintf('\nJoint-wise PCK:\n');
disp(pck * 100);

save( ...
    fullfile(basicDir,'baseline_same_eval_results.mat'), ...
    'results','apk','pck','prec','rec');

end