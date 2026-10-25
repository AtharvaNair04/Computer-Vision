function [out, info] = adaptive_preprocess(im, params)
% ADAPTIVE_PREPROCESS
% Stage 2 of the enhanced pipeline: quality-gated image restoration.
%
% Each image is analysed by image_quality() and at most ONE restoration is
% applied, in priority order (noise dominates HOG gradients most):
%
%   1. Noise   (noiseSigma > 6)       -> non-local-means denoising (imnlmfilt,
%                                        smoothing degree estimated per image)
%   2. Blur    (sharpness  < 0.014)   -> Richardson-Lucy deconvolution with a
%                                        Gaussian PSF (sigma 1.5, 8 iterations)
%   3. Dark    (brightness < 0.40)    -> gamma correction (gamma = 0.67)
%   otherwise                         -> image passed through unchanged
%
% Thresholds and operators were selected on PARSE test images 1-100 only
% (tuning/tune_report1.m, tuning/tune_gate.m); images 101-205 are held out.
% Operators that were tested and rejected on the same split: CLAHE-style
% contrast stretching, bilateral / median / Wiener / Gaussian denoising,
% unsharp masking and 1.25x upsampling.
%
% Output:
%   out  - restored image (same class as input)
%   info - quality measurements and the operation applied

    if nargin < 2, params = struct(); end
    params = set_default(params, 'noiseSigmaThresh', 6);
    params = set_default(params, 'sharpnessThresh', 0.014);
    params = set_default(params, 'brightnessThresh', 0.40);
    params = set_default(params, 'gamma', 0.67);
    params = set_default(params, 'deblurSigma', 1.5);
    params = set_default(params, 'deblurIters', 8);

    originalClass = class(im);
    if ~isa(im, 'uint8'), im = im2uint8(im); end

    q = image_quality(im);
    info.brightness = q.brightness;
    info.contrast   = q.contrast;
    info.noise      = q.noiseSigma;
    info.sharpness  = q.sharpness;
    info.operations = {};

    if q.noiseSigma > params.noiseSigmaThresh
        enhanced = imnlmfilt(im);
        info.operations{end+1} = 'Non-local-means denoising';
    elseif q.sharpness < params.sharpnessThresh
        psf = fspecial('gaussian', 9, params.deblurSigma);
        enhanced = im2uint8(deconvlucy(im2double(im), psf, params.deblurIters));
        info.operations{end+1} = 'Richardson-Lucy deblurring';
    elseif q.brightness < params.brightnessThresh
        enhanced = imadjust(im, [], [], params.gamma);
        info.operations{end+1} = 'Gamma correction';
    else
        enhanced = im;
    end
    info.numOperations = numel(info.operations);

    switch originalClass
        case 'double', out = im2double(enhanced);
        case 'single', out = im2single(enhanced);
        otherwise,     out = enhanced;
    end
end

function s = set_default(s, name, value)
    if ~isfield(s, name), s.(name) = value; end
end
