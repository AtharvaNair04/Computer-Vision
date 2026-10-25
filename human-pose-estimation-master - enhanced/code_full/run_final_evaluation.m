function results = run_final_evaluation(idx, varargin)
% RUN_FINAL_EVALUATION
% End-to-end evaluation of the enhanced FMP pipeline on PARSE
% (clean + dark / low-contrast / noise / blur degradations).
%
%   results = run_final_evaluation()            % held-out test images 101-205
%   results = run_final_evaluation(1:205)       % full PARSE test set
%   run_final_evaluation(101:205, 'shard', [k n])  % detection only, shard k of n
%                                                  % (run n MATLAB processes in
%                                                  %  parallel, then call again
%                                                  %  without 'shard' for metrics)
%
% PARSE test images 1-100 were used to tune every threshold and parameter
% (see tuning/). Report numbers on 101-205; 1:205 is given for reference.
%
% Ablation (each row adds one stage):
%   (1) Baseline FMP                     (Yang & Ramanan 2011)
%   (2) Adaptive HOG                     (adaptive_features, investigated)
%   (3) + Adaptive preprocessing         (quality-gated restoration)
%   (4) + Flip TTA + pose voting
%   (5) + Anatomical refinement          (full pipeline)
%
% Per-image detections are cached in cache/final/, so the run is resumable.

    if nargin < 1 || isempty(idx), idx = 101:205; end
    shard = [];
    for a = 1:2:numel(varargin)
        if strcmpi(varargin{a}, 'shard'), shard = varargin{a+1}; end
    end

    codeFullDir  = fileparts(mfilename('fullpath'));
    rootDir      = fileparts(codeFullDir);
    codeBasicDir = fullfile(rootDir, 'code_basic');
    figuresDir   = fullfile(rootDir, 'figures');
    cd(codeFullDir);
    addpath(codeFullDir, '-begin');
    addpath(fullfile(codeFullDir, 'detection'), '-begin');
    addpath(fullfile(codeFullDir, 'evaluation'), '-begin');
    addpath(fullfile(codeFullDir, 'visualization'), '-begin');
    addpath(codeBasicDir, '-end');                  % only for adaptive_detect_fast
    globals;

    S = load(fullfile(codeBasicDir, 'PARSE_model.mat'), 'model');
    model = S.model;
    model.thresh = min(model.thresh, -1);
    [~, test] = PARSE_data('PARSE');

    conds = {'clean', 'dark', 'lowcontrast', 'noise', 'blur'};
    cacheRoot = fullfile(codeFullDir, 'cache', 'final');

    %% ---------------------------------------------------------------
    % Stage A: detections (cached per image)
    % ---------------------------------------------------------------
    detIdx = idx;
    if ~isempty(shard), detIdx = idx(shard(1):shard(2):end); end
    for c = 1:numel(conds)
        dirC = fullfile(cacheRoot, conds{c});
        if ~exist(dirC, 'dir'), mkdir(dirC); end
        t0 = tic;
        for n = detIdx
            f = fullfile(dirC, sprintf('%03d.mat', n));
            if exist(f, 'file'), continue; end
            im = load_image(test, n, conds{c});
            D = struct();
            t = tic; D.base = detect_fast(im, model, model.thresh); D.tBase = toc(t);
            D.adapHOG = adaptive_detect_fast(im, model, model.thresh);
            [D.pre, infoP]  = enhanced_detect(im, model, struct('tta', false));
            [D.enh, infoE]  = enhanced_detect(im, model, struct('tta', true));
            D.ops = infoP.operations;
            D.tPre = infoE.timePreprocess;
            D.tEnh = infoE.timePreprocess + infoE.timeDetect;
            D = compact(D);
            save(f, '-struct', 'D');
        end
        fprintf('[%s] detections %-11s done (%d images, %.1f min)\n', ...
            datestr(now, 'HH:MM:SS'), conds{c}, numel(detIdx), toc(t0) / 60);
    end
    if ~isempty(shard)
        results = [];
        return;
    end

    %% ---------------------------------------------------------------
    % Stage B: metrics for every ablation row
    % ---------------------------------------------------------------
    methods = {'baseline', 'adaptiveHOG', 'preprocess', 'ttaVote', 'full'};
    names   = {'Baseline FMP', 'Adaptive HOG', '+ Preprocess', '+ Flip TTA/Vote', '+ Refinement'};
    mirror  = [6 5 4 3 2 1 12 11 10 9 8 7 14 13];
    order7  = [14 9 8 7 3 2 1];
    joints7 = {'Head', 'Shoulder', 'Elbow', 'Wrist', 'Hip', 'Knee', 'Ankle'};
    thresholds = 0.02:0.02:0.26;
    gt = test(idx);
    N = numel(idx);

    results = struct('idx', idx, 'conditions', {conds}, 'methods', {methods});
    timing = zeros(0, 3);
    for c = 1:numel(conds)
        det  = cell(1, numel(methods));
        pose = cell(1, numel(methods));
        [det{:}]  = deal(repmat(struct('obj', []), 1, N));
        [pose{:}] = deal(repmat(struct('obj', []), 1, N));
        opsCount = 0;
        for k = 1:N
            n = idx(k);
            D = load(fullfile(cacheRoot, conds{c}, sprintf('%03d.mat', n)));
            [pose{1}(k).obj, det{1}(k).obj] = select_plain(D.base, test(n).obj);
            [pose{2}(k).obj, det{2}(k).obj] = select_plain(D.adapHOG, test(n).obj);
            [pose{3}(k).obj, det{3}(k).obj] = select_plain(D.pre, test(n).obj);
            [pose{4}(k).obj, det{4}(k).obj] = select_vote(D.enh, test(n).obj);
            opsCount = opsCount + ~isempty(D.ops);
            timing(end+1, :) = [D.tBase, D.tPre, D.tEnh]; %#ok<AGROW>
        end
        for m = 1:4
            det{m}  = PARSE_transback(det{m});
            pose{m} = PARSE_transback(pose{m});
        end
        det{5}  = apply_pose_refinement(det{4});
        pose{5} = apply_pose_refinement(pose{4});

        for m = 1:numel(methods)
            r = struct();
            pck14 = eval_pck(pose{m}, gt, 0.1);
            apk14 = eval_apk(det{m}, gt, 0.1);
            r.pck7 = to7(pck14, mirror, order7);
            r.apk7 = to7(apk14, mirror, order7);
            r.meanPCK = mean(r.pck7);
            r.meanAPK = mean(r.apk7);
            r.pckCurve = arrayfun(@(t) mean(eval_pck(pose{m}, gt, t)), thresholds);
            r.pckPerImage = per_image_pck(pose{m}, gt);
            results.(conds{c}).(methods{m}) = r;
        end
        base = results.(conds{c}).baseline.pckPerImage;
        full = results.(conds{c}).full.pckPerImage;
        results.(conds{c}).deltaCI = bootstrap_ci(full - base);
        results.(conds{c}).fracPreprocessed = opsCount / N;
        results.(conds{c}).posesFull = pose{5};
        results.(conds{c}).posesBase = pose{1};
    end
    results.timing.baselineMs   = 1000 * mean(timing(:, 1));
    results.timing.preprocessMs = 1000 * mean(timing(:, 2));
    results.timing.enhancedMs   = 1000 * mean(timing(:, 3));

    print_report(results, conds, methods, names, joints7, idx);
    save(fullfile(codeFullDir, sprintf('final_results_%d_%d.mat', idx(1), idx(end))), 'results');

    if ~exist(figuresDir, 'dir'), mkdir(figuresDir); end
    make_figures(results, conds, methods, names, joints7, thresholds, test, figuresDir);
end

%% ===================================================================
function im = load_image(test, n, cond)
    if strcmp(cond, 'clean')
        im = imread(test(n).im);
    else
        im = imread(fullfile('data', 'PARSE_degraded', cond, sprintf('im%04d.jpg', 100 + n)));
    end
end

function D = compact(D)
% Keep the 500 best candidates of each method (single precision) on disk.
    f = {'base', 'adapHOG', 'pre'};
    for i = 1:numel(f)
        b = D.(f{i});
        if size(b, 1) > 500
            [~, I] = sort(b(:, end), 'descend');
            b = b(I(1:500), :);
        end
        D.(f{i}) = single(b);
    end
    D.enh = single(D.enh);
end

function [poseObj, detObj] = select_plain(boxes, gtObj)
% Official protocol: PCK uses the best-scoring detection overlapping the
% annotated person; APK uses all detections after NMS.
    boxes = double(boxes);
    poseObj = []; detObj = [];
    if isempty(boxes), return; end
    b = bestoverlap(boxes, gtObj, 0.3);
    if ~isempty(b), poseObj = box2obj(b); end
    detObj = box2obj(nms(boxes, 0.3));
end

function [poseObj, detObj] = select_vote(boxes, gtObj)
% Same protocol, with pose voting applied to every selected detection.
    boxes = double(boxes);
    poseObj = []; detObj = [];
    if isempty(boxes), return; end
    b = bestoverlap(boxes, gtObj, 0.3);
    if ~isempty(b), poseObj = pose_vote(b, boxes); end
    top = nms(boxes, 0.3);
    for t = 1:size(top, 1)
        detObj = [detObj, pose_vote(top(t, :), boxes)]; %#ok<AGROW>
    end
end

function v7 = to7(v14, mirror, order7)
    v = (v14 + v14(mirror)) / 2;
    v7 = v(order7);
end

function p = per_image_pck(pose14, gt)
    p = zeros(1, numel(gt));
    for n = 1:numel(gt)
        if isempty(pose14(n).obj) || isempty(pose14(n).obj(1).point), continue; end
        pts = gt(n).obj(1).point;
        sc = max(max(pts) - min(pts) + 1);
        d = sqrt(sum((pose14(n).obj(1).point - pts) .^ 2, 2));
        p(n) = mean(d <= 0.1 * sc);
    end
end

function ci = bootstrap_ci(d)
% Paired bootstrap 95% CI of the mean per-image PCK change.
    rng(0);
    B = 5000;
    m = zeros(B, 1);
    for b = 1:B
        m(b) = mean(d(randi(numel(d), numel(d), 1)));
    end
    ci = [mean(d), prctile(m, 2.5), prctile(m, 97.5)];
end

function print_report(R, conds, methods, names, joints7, idx)
    fprintf('\n=====================================================================================\n');
    fprintf(' PARSE RESULTS  (test images %d-%d, N = %d)\n', idx(1), idx(end), numel(idx));
    fprintf('=====================================================================================\n');
    fprintf('%-12s', 'Condition');
    for m = 1:numel(methods), fprintf(' | %-15s', names{m}); end
    fprintf(' | Full - Base PCK [95%% CI]\n');
    for metric = {'meanPCK', 'meanAPK'}
        fprintf('---- %s ----\n', strrep(metric{1}, 'mean', ''));
        for c = 1:numel(conds)
            fprintf('%-12s', conds{c});
            for m = 1:numel(methods)
                fprintf(' | %15.2f', 100 * R.(conds{c}).(methods{m}).(metric{1}));
            end
            if strcmp(metric{1}, 'meanPCK')
                ci = 100 * R.(conds{c}).deltaCI;
                fprintf(' | %+5.2f [%+5.2f, %+5.2f]', ci);
            end
            fprintf('\n');
        end
    end
    fprintf('\nPer-joint PCK, clean, Baseline -> Full:\n');
    for j = 1:7
        fprintf('  %-9s %5.1f -> %5.1f\n', joints7{j}, 100 * R.clean.baseline.pck7(j), 100 * R.clean.full.pck7(j));
    end
    fprintf('\nFraction of images restored by the quality gate:');
    for c = 1:numel(conds), fprintf('  %s %.0f%%', conds{c}, 100 * R.(conds{c}).fracPreprocessed); end
    fprintf('\nMean latency per image: baseline %.0f ms | enhanced %.0f ms (of which preprocessing %.0f ms)\n\n', ...
        R.timing.baselineMs, R.timing.enhancedMs, R.timing.preprocessMs);
end

function make_figures(R, conds, methods, names, joints7, thresholds, test, figuresDir)
    for metric = {'meanPCK', 'meanAPK'}
        M = zeros(numel(conds), numel(methods));
        for c = 1:numel(conds)
            for m = 1:numel(methods)
                M(c, m) = 100 * R.(conds{c}).(methods{m}).(metric{1});
            end
        end
        f = new_figure('Position', [100 100 1000 420]);
        bar(M);
        set(gca, 'XTickLabel', conds, 'FontSize', 11);
        ylabel(sprintf('%s (%%)', strrep(metric{1}, 'mean', '')));
        ylim([max(0, min(M(:)) - 10), min(100, max(M(:)) + 5)]);
        legend(names, 'Location', 'southoutside', 'Orientation', 'horizontal');
        title(sprintf('PARSE %s by condition (test %d-%d)', strrep(metric{1}, 'mean', ''), R.idx(1), R.idx(end)));
        grid on;
        saveas(f, fullfile(figuresDir, sprintf('final_%s_by_condition.png', lower(strrep(metric{1}, 'mean', '')))));
        close(f);
    end

    f = new_figure('Position', [100 100 800 420]);
    bar(100 * [R.clean.baseline.pck7; R.clean.full.pck7]');
    set(gca, 'XTickLabel', joints7, 'FontSize', 11);
    ylabel('PCK (%)'); legend({'Baseline FMP', 'Full pipeline'}, 'Location', 'northeast');
    title('Per-joint PCK, clean PARSE'); grid on;
    saveas(f, fullfile(figuresDir, 'final_per_joint_clean.png'));
    close(f);

    f = new_figure('Position', [100 100 1200 360]);
    for c = 1:numel(conds)
        subplot(1, numel(conds), c);
        plot(thresholds, 100 * R.(conds{c}).baseline.pckCurve, 'k--', 'LineWidth', 1.5); hold on;
        plot(thresholds, 100 * R.(conds{c}).full.pckCurve, 'r-', 'LineWidth', 1.5);
        title(conds{c}); xlabel('threshold'); if c == 1, ylabel('PCK (%)'); end
        grid on; ylim([0 100]);
    end
    legend({'Baseline', 'Full'}, 'Location', 'southeast');
    saveas(f, fullfile(figuresDir, 'final_pck_curves.png'));
    close(f);

    % Qualitative: image where the full pipeline gains most, per condition
    f = new_figure('Position', [50 50 1400 600]);
    for c = 1:numel(conds)
        d = R.(conds{c}).full.pckPerImage - R.(conds{c}).baseline.pckPerImage;
        [~, k] = max(d);
        n = R.idx(k);
        im = load_image(test, n, conds{c});
        pb = R.(conds{c}).posesBase(k).obj; pf = R.(conds{c}).posesFull(k).obj;
        subplot(2, numel(conds), c);
        if ~isempty(pb), plot_parse_skeleton(im, pb(1).point, struct('title', sprintf('%s: baseline', conds{c}))); else, imshow(im); title(sprintf('%s: baseline (no detection)', conds{c})); end
        subplot(2, numel(conds), numel(conds) + c);
        if ~isempty(pf), plot_parse_skeleton(im, pf(1).point, struct('title', 'full pipeline')); else, imshow(im); title('full pipeline (no detection)'); end
    end
    saveas(f, fullfile(figuresDir, 'final_qualitative.png'));
    close(f);
end

function f = new_figure(varargin)
% Invisible figure with a light theme (MATLAB R2025a+ defaults to the
% desktop theme, which makes saved titles unreadable).
    f = figure('Visible', 'off', varargin{:});
    try
        theme(f, 'light');
    catch
        set(f, 'Color', 'w');
    end
end
