function [out, info] = adaptive_preprocess(im)
% ADAPTIVE_PREPROCESS
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
    if brightness < 0.30

        % Adaptive gamma correction
        gamma = max(0.55, min(0.85, brightness / 0.30));

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
    % ---------------------------------------------------------
    if contrast < 0.18

        if size(enhanced, 3) == 3

            % Convert to L*a*b colour space
            lab = rgb2lab(enhanced);

            L = lab(:,:,1) / 100;

            % CLAHE
            L = adapthisteq( ...
                L, ...
                'NumTiles', [8 8], ...
                'ClipLimit', 0.015 ...
            );

            lab(:,:,1) = L * 100;

            enhanced = lab2rgb( ...
                lab, ...
                'OutputType', ...
                'uint8' ...
            );

        else

            enhanced = adapthisteq( ...
                enhanced, ...
                'NumTiles', [8 8], ...
                'ClipLimit', 0.015 ...
            );

        end

        info.operations{end+1} = 'CLAHE';

    end

    % ---------------------------------------------------------
    % C. High noise
    % ---------------------------------------------------------
    if noiseLevel > 0.055

        enhanced = imbilatfilt( ...
            enhanced, ...
            25, ...
            7 ...
        );

        info.operations{end+1} = 'Bilateral denoising';

    end

    % ---------------------------------------------------------
    % D. Low sharpness
    % ---------------------------------------------------------
    if sharpness < 0.002

        enhanced = imsharpen( ...
            enhanced, ...
            'Radius', 1, ...
            'Amount', 0.8 ...
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