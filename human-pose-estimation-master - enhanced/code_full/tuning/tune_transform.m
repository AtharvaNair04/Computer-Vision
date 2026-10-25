function [out, scale, flipped] = tune_transform(im, variant)
% TUNE_TRANSFORM
% Image-level transform for a named tuning variant.
% A trailing '_flip' mirrors the image horizontally after the transform.
%
% Returns the transformed image, the resize factor applied (boxes must be
% divided by it), and whether the image was mirrored.

    flipped = false;
    if numel(variant) > 5 && strcmp(variant(end-4:end), '_flip')
        flipped = true;
        variant = variant(1:end-5);
    end

    if ~isa(im, 'uint8'), im = im2uint8(im); end
    scale = 1;

    switch variant
        case 'base'
            out = im;
        case 'adapt_v1'
            out = adaptive_preprocess_v1(im);        % pre-tuning repo pipeline
        case 'gauss07'
            out = imgaussfilt(im, 0.7);
        case 'gauss10'
            out = imgaussfilt(im, 1.0);
        case 'median3'
            out = medfilt_rgb(im, 3);
        case 'nlm'
            out = imnlmfilt(im);                      % auto-estimated smoothing
        case 'bilat_strong'
            out = imbilatfilt(im, 2 * estimate_noise_var(im), 1.5);
        case 'wiener5'
            out = wiener_rgb(im, 5);
        case 'gamma067'
            out = imadjust(im, [], [], 0.67);
        case 'stretch'
            out = imadjust(im, stretchlim(im, [0.005 0.995]), []);
        case 'deconv15'
            out = im2uint8(deconvlucy(im2double(im), fspecial('gaussian', 9, 1.5), 8));
        case 'sharp_mild'
            out = imsharpen(im, 'Radius', 1.5, 'Amount', 0.6);
        case 'up125'
            scale = 1.25;
            out = imresize(im, scale, 'bicubic');
        otherwise
            error('Unknown variant: %s', variant);
    end

    if flipped
        out = out(:, end:-1:1, :);
    end
end

function out = medfilt_rgb(im, k)
    out = im;
    for c = 1:size(im, 3)
        out(:,:,c) = medfilt2(im(:,:,c), [k k], 'symmetric');
    end
end

function out = wiener_rgb(im, k)
    out = im;
    for c = 1:size(im, 3)
        out(:,:,c) = wiener2(im(:,:,c), [k k]);
    end
end

function v = estimate_noise_var(im)
% Immerkaer (1996) fast noise variance estimate on the uint8 scale.
    g = double(rgb2gray_safe(im));
    M = [1 -2 1; -2 4 -2; 1 -2 1];
    [H, W] = size(g);
    s = sum(sum(abs(conv2(g, M, 'valid'))));
    sigma = s * sqrt(pi / 2) / (6 * (W - 2) * (H - 2));
    v = sigma ^ 2;
end

function g = rgb2gray_safe(im)
    if size(im, 3) == 3, g = rgb2gray(im); else, g = im; end
end
