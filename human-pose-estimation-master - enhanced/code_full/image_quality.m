function q = image_quality(im)
% IMAGE_QUALITY
% Stage 1 of the enhanced pipeline: no-reference image-quality analysis.
%
%   brightness - mean luminance in [0,1]
%   contrast   - robust luminance range (1st to 99th percentile) in [0,1]
%   noiseSigma - Gaussian noise std on the 0-255 scale, Immerkaer (1996)
%                fast estimator: a Laplacian-difference mask that cancels
%                image structure up to second order
%   sharpness  - Laplacian energy relative to luminance variance; this is
%                contrast-invariant, so low-contrast images are not
%                mistaken for blurred ones

    if ~isa(im, 'uint8'), im = im2uint8(im); end
    if size(im, 3) == 3, g = rgb2gray(im); else, g = im; end
    g = double(g);
    [H, W] = size(g);

    q.brightness = mean(g(:)) / 255;
    p = prctile(g(:), [1 99]);
    q.contrast = (p(2) - p(1)) / 255;

    M = [1 -2 1; -2 4 -2; 1 -2 1];
    q.noiseSigma = sum(sum(abs(conv2(g, M, 'valid')))) * sqrt(pi / 2) / (6 * (W - 2) * (H - 2));

    % Smooth away sensor noise first so noisy images do not look "sharp"
    gs  = imgaussfilt(g, 0.8);
    lap = conv2(gs, [0 1 0; 1 -4 1; 0 1 0], 'valid');
    q.sharpness = var(lap(:)) / max(var(g(:)), 1);
end
