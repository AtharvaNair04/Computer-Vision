function boxes = detect_tta(im, model, thresh, topK)
% DETECT_TTA
% Stage 4 with flip test-time augmentation: runs the unchanged FMP detector
% on the image and on its mirror image, maps the mirrored detections back
% (x-coordinates reflected, left/right part identities swapped) and returns
% the pooled candidate set. Each pass keeps its topK highest-scoring
% candidates (default 500).
%
% Part mirror map: FMP trees are [head, neck, side A (n-2)/2 parts,
% side B (n-2)/2 parts], so mirroring swaps the two halves
% (PARSE: [1 2 15:26 3:14], BUFFY: [1 2 11:18 3:10]).

    if nargin < 4, topK = 500; end

    boxes  = keep_top(detect_fast(im, model, thresh), topK);
    boxesF = keep_top(detect_fast(im(:, end:-1:1, :), model, thresh), topK);
    if isempty(boxesF), return; end

    numparts = floor((size(boxesF, 2) - 2) / 5);
    half = (numparts - 2) / 2;
    mirror = [1 2, (3 + half):numparts, 3:(2 + half)];

    W = size(im, 2);
    for p = 1:numparts
        c = 5 * (p - 1);
        x1 = W + 1 - boxesF(:, c + 3);
        x2 = W + 1 - boxesF(:, c + 1);
        boxesF(:, c + 1) = x1;
        boxesF(:, c + 3) = x2;
    end
    cols = reshape(1:5 * numparts, 5, numparts);
    cols = cols(:, mirror);
    boxesF(:, 1:5 * numparts) = boxesF(:, cols(:)');

    boxes = [boxes; boxesF];
end

function b = keep_top(b, K)
    if size(b, 1) > K
        [~, I] = sort(b(:, end), 'descend');
        b = b(I(1:K), :);
    end
end
