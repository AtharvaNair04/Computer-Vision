function [res, detP, poseP] = tune_eval(spec)
% TUNE_EVAL
% Scores cached tuning detections with the official PCK / APK protocol.
%
% spec fields:
%   variants - cell array of variants whose detections are pooled
%   perImage - optional cell (one per idx) overriding variants per image
%   cond     - degradation condition
%   idx      - test indices
%   vote     - [] or struct(iou, delta, T) for pose voting
%   refine   - true to apply refine_pose on the final 14 joints
%   refineParams - optional params for refine_pose

    persistent test
    codeFullDir = fileparts(fileparts(mfilename('fullpath')));
    if isempty(test)
        old = cd(codeFullDir);
        addpath(codeFullDir, fullfile(codeFullDir, 'detection'), fullfile(codeFullDir, 'evaluation'));
        globals;
        [~, test] = PARSE_data('PARSE');
        cd(old);
    end
    if ~isfield(spec, 'vote'), spec.vote = []; end
    if ~isfield(spec, 'refine'), spec.refine = false; end
    if ~isfield(spec, 'refineParams'), spec.refineParams = struct(); end

    N = numel(spec.idx);
    detP  = repmat(struct('obj', []), 1, N);
    poseP = repmat(struct('obj', []), 1, N);
    for k = 1:N
        n = spec.idx(k);
        boxes = [];
        variants = spec.variants;
        if isfield(spec, 'perImage'), variants = spec.perImage{k}; end
        for v = 1:numel(variants)
            f = fullfile(codeFullDir, 'cache', 'tuning', variants{v}, spec.cond, sprintf('%03d.mat', n));
            S = load(f, 'boxes');
            boxes = [boxes; double(S.boxes)]; %#ok<AGROW>
        end
        if isempty(boxes), continue; end

        best = bestoverlap(boxes, test(n).obj, 0.3);
        if ~isempty(best)
            poseP(k).obj = vote_obj(best, boxes, spec.vote);
        end
        top = nms(boxes, 0.3);
        objs = [];
        for t = 1:size(top, 1)
            objs = [objs, vote_obj(top(t, :), boxes, spec.vote)]; %#ok<AGROW>
        end
        detP(k).obj = objs;
    end

    det14  = PARSE_transback(detP);
    pose14 = PARSE_transback(poseP);
    if spec.refine
        det14  = apply_pose_refinement(det14, spec.refineParams);
        pose14 = apply_pose_refinement(pose14, spec.refineParams);
    end

    gt = test(spec.idx);
    mirror = [6 5 4 3 2 1 12 11 10 9 8 7 14 13];
    order7 = [14 9 8 7 3 2 1];
    pck14 = eval_pck(pose14, gt, 0.1);
    apk14 = eval_apk(det14, gt, 0.1);
    res.pck7 = (pck14(order7) + pck14(mirror(order7))) / 2;
    res.apk7 = (apk14(order7) + apk14(mirror(order7))) / 2;
    res.pck = mean(res.pck7);
    res.apk = mean(res.apk7);
    res.pckPerImage = per_image_pck(pose14, gt);
end

function obj = vote_obj(box, boxes, vote)
    obj = box2obj(box);
    if isempty(vote), return; end
    iou = person_iou(box, boxes);
    s = boxes(:, end);
    keep = iou > vote.iou & s >= box(end) - vote.delta;
    if nnz(keep) < 2, return; end
    w = exp((s(keep) - box(end)) / vote.T);
    w = w / sum(w);
    cand = box2obj(boxes(keep, :));
    pts = zeros(size(obj.point));
    for i = 1:numel(cand)
        pts = pts + w(i) * cand(i).point;
    end
    obj.point = pts;
end

function o = person_iou(box, boxes)
    numpart = floor(size(boxes, 2) / 5);
    idx = 5 * ((1:numpart) - 1);
    b  = [min(box(idx+1)) min(box(idx+2)) max(box(idx+3)) max(box(idx+4))];
    bx = [min(boxes(:,idx+1),[],2) min(boxes(:,idx+2),[],2) max(boxes(:,idx+3),[],2) max(boxes(:,idx+4),[],2)];
    w = min(b(3), bx(:,3)) - max(b(1), bx(:,1)) + 1; w(w < 0) = 0;
    h = min(b(4), bx(:,4)) - max(b(2), bx(:,2)) + 1; h(h < 0) = 0;
    inter = w .* h;
    a  = (b(3) - b(1) + 1) * (b(4) - b(2) + 1);
    ax = (bx(:,3) - bx(:,1) + 1) .* (bx(:,4) - bx(:,2) + 1);
    o = inter ./ (a + ax - inter);
end

function p = per_image_pck(pose14, gt)
    p = zeros(1, numel(gt));
    for n = 1:numel(gt)
        if isempty(pose14(n).obj) || isempty(pose14(n).obj(1).point), continue; end
        pts = gt(n).obj(1).point;
        sc = max(max(pts) - min(pts) + 1);
        d = sqrt(sum((pose14(n).obj(1).point - pts).^2, 2));
        p(n) = mean(d <= 0.1 * sc);
    end
end
