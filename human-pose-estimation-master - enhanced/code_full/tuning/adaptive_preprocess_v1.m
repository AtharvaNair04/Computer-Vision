function [out, info] = adaptive_preprocess_v1(im)
% ADAPTIVE_PREPROCESS_V1 (pre-tuning version, kept for the ablation)
% Adaptive image enhancement for Flexible Mixtures-of-Parts.
%
% Input:
%   im   - RGB or grayscale image
%
% Output:
%   out  - enhanced image
%   info - image quality measurements and applied operations
%
% The module adapts preprocessing according to:
%   1. Brightness
%   2. Contrast
%   3. Estimated noise
%   4. Estimated sharpness

    % ---------------------------------------------------------
    % Preserve original image type
    % ---------------------------------------------------------
    originalClass = class(im);

    if ~isa(im, 'uint8')
        im = im2uint8(im);
    end

    % ---------------------------------------------------------
    % Convert to grayscale for quality analysis
    % ---------------------------------------------------------
    if size(im, 3) == 3
        gray = rgb2gray(im);
    else
        gray = im;
    end

    grayDouble = im2double(gray);

    % ---------------------------------------------------------
    % 1. Brightness
    % ---------------------------------------------------------
    brightness = mean(grayDouble(:));

    % ---------------------------------------------------------
    % 2. Contrast
    % ---------------------------------------------------------
    contrast = std(grayDouble(:));

    % ---------------------------------------------------------
    % 3. Estimate noise
    %
    % High-frequency residual:
    % image - smoothed image
    % ---------------------------------------------------------
    smoothImage = imgaussfilt(grayDouble, 1);

    residual = grayDouble - smoothImage;

    noiseLevel = std(residual(:));

    % ---------------------------------------------------------
    % 4. Estimate sharpness
    %
    % Variance of Laplacian
    % ---------------------------------------------------------
    laplacianKernel = [0 1 0;
                       1 -4 1;
                       0 1 0];

    laplacianResponse = imfilter( ...
        grayDouble, ...
        laplacianKernel, ...
        'replicate' ...
    );

    sharpness = var(laplacianResponse(:));

    % ---------------------------------------------------------
    % Store quality measurements
    % ---------------------------------------------------------
    info.brightness = brightness;
    info.contrast = contrast;
    info.noise = noiseLevel;
    info.sharpness = sharpness;

    info.operations = {};

    enhanced = im;

    % =========================================================
    % ADAPTIVE ENHANCEMENT
    % =========================================================

    % ---------------------------------------------------------
    % A. Very dark image
    % ---------------------------------------------------------
    if brightness < 0.12

        % Adaptive gamma correction
        gamma = max(0.60, min(0.85, brightness / 0.15));

        enhanced = imadjust( ...
            enhanced, ...
            [], ...
            [], ...
            gamma ...
        );

        info.operations{end+1} = 'Gamma correction';

    end

    % ---------------------------------------------------------
    % B. Low contrast image
    %
    % HOG features employ local L2-Hys block normalization which
    % natively compensates for global contrast scaling.
    % Artificially stretching pixel contrast (e.g. CLAHE) introduces
    % spurious tile-boundary gradient artifacts and amplifies clutter.
    % We preserve raw gradients for low-contrast images unless dark.
    % ---------------------------------------------------------
    if contrast < 0.08 && brightness < 0.20
        % Only apply gentle linear adjustment when severely dim & low contrast
        enhanced = imadjust(enhanced, stretchlim(enhanced, [0.01 0.99]), []);
        info.operations{end+1} = 'Gentle contrast normalization';
    end

    % ---------------------------------------------------------
    % C. High noise
    % ---------------------------------------------------------
    if noiseLevel > 0.070

        enhanced = imbilatfilt( ...
            enhanced, ...
            400, ...
            1.2 ...
        );

        info.operations{end+1} = 'Bilateral denoising';

    end

    % ---------------------------------------------------------
    % D. Low sharpness (Blur)
    % ---------------------------------------------------------
    if sharpness < 0.0012

        enhanced = imsharpen( ...
            enhanced, ...
            'Radius', 1, ...
            'Amount', 0.45, ...
            'Threshold', 0.015 ...
        );

        info.operations{end+1} = 'Detail enhancement';

    end

    % ---------------------------------------------------------
    % Safety check
    % ---------------------------------------------------------
    enhanced = min(max(enhanced, 0), 255);

    % ---------------------------------------------------------
    % Restore original data type
    % ---------------------------------------------------------
    if strcmp(originalClass, 'double')

        out = im2double(enhanced);

    elseif strcmp(originalClass, 'single')

        out = im2single(enhanced);

    else

        out = uint8(enhanced);

    end

    % ---------------------------------------------------------
    % Final statistics
    % ---------------------------------------------------------
    info.numOperations = length(info.operations);

end