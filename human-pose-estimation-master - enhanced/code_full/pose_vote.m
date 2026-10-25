function obj = pose_vote(box, boxes, params)
% POSE_VOTE
% Stage 4b: score-weighted pose voting. Instead of trusting the single
% arg-max FMP configuration, every part location is replaced by the
% weighted mean over near-duplicate candidates of the same person:
%
%   candidates : person-box IoU with the selected detection > params.iou
%                and score >= s* - params.delta
%   weights    : w_i = exp((s_i - s*) / params.T)
%
% This averages out the quantisation and mixture-switching jitter of the
% HOG grid (and, with flip TTA, of the two mirrored passes).
% Defaults (iou 0.7, delta 0.25, T 0.1) were chosen on PARSE test 1-100.
%
% Returns a box2obj-style struct for the selected detection.

    if nargin < 3, params = struct(); end
    if ~isfield(params, 'iou'),   params.iou = 0.7;    end
    if ~isfield(params, 'delta'), params.delta = 0.25; end
    if ~isfield(params, 'T'),     params.T = 0.1;      end

    obj = box2obj(box);
    s = boxes(:, end);
    keep = person_iou(box, boxes) > params.iou & s >= box(end) - params.delta;
    if nnz(keep) < 2, return; end

    w = exp((s(keep) - box(end)) / params.T);
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
    bx = [min(boxes(:,idx+1),[],2) min(boxes(:,idx+2),[],2) ...
          max(boxes(:,idx+3),[],2) max(boxes(:,idx+4),[],2)];
    w = min(b(3), bx(:,3)) - max(b(1), bx(:,1)) + 1; w(w < 0) = 0;
    h = min(b(4), bx(:,4)) - max(b(2), bx(:,2)) + 1; h(h < 0) = 0;
    inter = w .* h;
    a  = (b(3) - b(1) + 1) * (b(4) - b(2) + 1);
    ax = (bx(:,3) - bx(:,1) + 1) .* (bx(:,4) - bx(:,2) + 1);
    o = inter ./ (a + ax - inter);
end
