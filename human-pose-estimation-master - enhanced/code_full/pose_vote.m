function obj = pose_vote(box, boxes, params)
% POSE_VOTE
% Stage 4b: part-consistent, score-weighted pose voting. Instead of
% trusting the single arg-max FMP configuration, each part location is
% replaced by the weighted mean over near-duplicate candidates of the same
% person that AGREE on that part:
%
%   candidates : person-box IoU with the selected detection > params.iou
%                and score >= s* - params.delta
%   per part j : a candidate votes only if its part j lies within
%                params.r x (selected person size) of the selected part j,
%                so different limb hypotheses are never averaged together
%   weights    : w_i = exp((s_i - s*) / params.T)
%
% This averages out the quantisation and mixture-switching jitter of the
% HOG grid (and, with flip TTA, of the two mirrored passes).
% Defaults (iou 0.5, delta 1.0, T 0.5, r 0.25) were chosen on PARSE test
% images 1-100 (tuning/tune_vote2.m).
%
% Returns a box2obj-style struct for the selected detection.

    if nargin < 3, params = struct(); end
    if ~isfield(params, 'iou'),   params.iou = 0.5;   end
    if ~isfield(params, 'delta'), params.delta = 1.0; end
    if ~isfield(params, 'T'),     params.T = 0.5;     end
    if ~isfield(params, 'r'),     params.r = 0.25;    end

    obj = box2obj(box);
    s = boxes(:, end);
    keep = person_iou(box, boxes) > params.iou & s >= box(end) - params.delta;
    if nnz(keep) < 2, return; end

    w = exp((s(keep) - box(end)) / params.T);
    cand = box2obj(boxes(keep, :));
    P = cat(3, cand.point);                                   % parts x 2 x K
    W = repmat(reshape(w, 1, 1, []), size(P, 1), 1, 1);       % parts x 1 x K
    if isfinite(params.r)
        sz = max(max(obj.point) - min(obj.point));
        d = sqrt(sum((P - obj.point) .^ 2, 2));
        W = W .* (d <= params.r * sz);                         % the selected box always votes
    end
    obj.point = sum(P .* W, 3) ./ sum(W, 3);
end

function o = person_iou(box, boxes)
    numpart = floor(size(boxes, 2) / 5);
    idx = 5 * ((1:numpart) - 1);
    b  = [min(box(idx+1)) min(box(idx+2)) max(box(idx+3)) max(box(idx+4))];
    bx = [min(boxes(:,idx+1),[],2) min(boxes(:,idx+2),[],2) ...
          max(boxes(:,idx+3),[],2) max(boxes(:,idx+4),[],2)];
    w = min(b(3), bx(:,3)) - max(b(1), bx(:,1)) + 1; w(w < 0) = 0;
    h = min(b(4), bx(:,4)) - max(b(2), bx(:,2)) + 1; h(h < 0) = 0;
    inter = w .* h;
    a  = (b(3) - b(1) + 1) * (b(4) - b(2) + 1);
    ax = (bx(:,3) - bx(:,1) + 1) .* (bx(:,4) - bx(:,2) + 1);
    o = inter ./ (a + ax - inter);
end
