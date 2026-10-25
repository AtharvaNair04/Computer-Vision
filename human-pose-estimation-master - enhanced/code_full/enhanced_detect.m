function [boxes, info] = enhanced_detect(im, model, opts)
% ENHANCED_DETECT
% Stages 1-4 of the enhanced FMP pipeline:
%   image-quality analysis -> gated restoration (adaptive_preprocess)
%   -> unchanged FMP inference with flip test-time augmentation (detect_tta)
%
% Returns the pooled candidate boxes in original image coordinates.
% Use pose_vote() on the selected box(es), then the dataset transback and
% apply_pose_refinement() for the remaining stages.
%
% opts.preprocess (default true), opts.tta (default true)

    if nargin < 3, opts = struct(); end
    if ~isfield(opts, 'preprocess'), opts.preprocess = true; end
    if ~isfield(opts, 'tta'), opts.tta = true; end

    t = tic;
    if opts.preprocess
        [imIn, info] = adaptive_preprocess(im);
    else
        imIn = im;
        info = struct('operations', {{}}, 'numOperations', 0);
    end
    info.timePreprocess = toc(t);

    t = tic;
    if opts.tta
        boxes = detect_tta(imIn, model, model.thresh);
    else
        boxes = detect_fast(imIn, model, model.thresh);
    end
    info.timeDetect = toc(t);
end
