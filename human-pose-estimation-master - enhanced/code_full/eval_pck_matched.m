function pck = eval_pck_matched(det, gt, thresh)

if nargin < 3
    thresh = 0.1;
end

assert(numel(det) == numel(gt));

numpart = size(gt(1).obj(1).point, 1);
tp = zeros(numpart, 0);

for n = 1:numel(gt)

    % Ground-truth scale
    point = gt(n).obj(1).point;

    box = [min(point, [], 1) max(point, [], 1)];

    scale = max( ...
        box(3) - box(1) + 1, ...
        box(4) - box(2) + 1);

    % No detection for this image
    if isempty(det(n).obj)

        tp(:, end+1) = false(numpart, 1);
        continue;

    end

    % PARSE has one ground-truth person per image.
    % Use the highest-scoring detection.
    scores = [det(n).obj.score];
    [~, best] = max(scores);

    detectedPoint = det(n).obj(best).point;

    % Distance for every keypoint
    dist = sqrt(sum( ...
        (detectedPoint - point).^2, 2));

    % Correct if within threshold * person scale
    tp(:, end+1) = ...
        dist <= thresh * scale;

end

% PCK for each keypoint
pck = mean(tp, 2)';

end