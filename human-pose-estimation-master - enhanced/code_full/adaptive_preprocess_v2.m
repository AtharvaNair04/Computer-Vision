function [out, info] = adaptive_preprocess_v2(im)
% ADAPTIVE_PREPROCESS_V2
% Conservative adaptive preprocessing for FMP/HOG.
%
% The goal is to improve poor-quality images while preserving
% the local gradients used by the original FMP model.

    originalClass = class(im);

    if ~isa(im, 'uint8')
        im = im2uint8(im);
    end

    % ---------------------------------------------------------
    % Quality analysis
    % ---------------------------------------------------------

    if size(im,3) == 3
        gray = rgb2gray(im);
    else
        gray = im;
    end

    grayDouble = im2double(gray);

    brightness = mean(grayDouble(:));
    contrast = std(grayDouble(:));

    smoothImage = imgaussfilt(grayDouble, 1);
    residual = grayDouble - smoothImage;
    noiseLevel = std(residual(:));

    laplacianKernel = [0 1 0;
                       1 -4 1;
                       0 1 0];

    laplacianResponse = imfilter( ...
        grayDouble, ...
        laplacianKernel, ...
        'replicate');

    sharpness = var(laplacianResponse(:));

    % ---------------------------------------------------------
    % Store measurements
    % ---------------------------------------------------------

    info.brightness = brightness;
    info.contrast = contrast;
    info.noise = noiseLevel;
    info.sharpness = sharpness;
    info.operations = {};

    enhanced = im;

    % ---------------------------------------------------------
    % 1. Mild gamma correction
    % ---------------------------------------------------------
    
    if brightness < 0.25

        % Mild correction only
        gamma = 0.85;

        enhanced = imadjust( ...
            enhanced, ...
            [], ...
            [], ...
            gamma);

        info.operations{end+1} = 'Mild gamma correction';

    elseif brightness > 0.80

        % Mild correction for very bright images
        gamma = 1.10;

        enhanced = imadjust( ...
            enhanced, ...
            [], ...
            [], ...
            gamma);

        info.operations{end+1} = 'Mild brightness correction';

    end

    % ---------------------------------------------------------
    % 2. Mild global contrast enhancement
    % ---------------------------------------------------------

    if contrast < 0.12

        % Avoid CLAHE because it can strongly modify HOG gradients.

        if size(enhanced,3) == 3

            lab = rgb2lab(enhanced);

            L = lab(:,:,1) / 100;

            % Gentle contrast stretch
            low = min(L(:));
            high = max(L(:));

            if high > low
                L = (L - low) / (high - low);
            end

            lab(:,:,1) = L * 100;

            enhanced = lab2rgb( ...
                lab, ...
                'OutputType', ...
                'uint8');

        else

            low = double(min(enhanced(:)));
            high = double(max(enhanced(:)));

            if high > low
                enhanced = uint8( ...
                    255 * ...
                    (double(enhanced) - low) / ...
                    (high - low));
            end

        end

        info.operations{end+1} = 'Gentle contrast enhancement';

    end

    % ---------------------------------------------------------
    % 3. Do NOT denoise by default
    % ---------------------------------------------------------
    %
    % HOG relies on local gradients. Strong denoising can remove
    % useful body-part boundaries.
    %
    % Therefore noise is measured but not modified in V2.

    if noiseLevel > 0.08
        info.operations{end+1} = 'High noise detected - unchanged';
    end

    % ---------------------------------------------------------
    % 4. Do NOT sharpen by default
    % ---------------------------------------------------------
    %
    % Artificial sharpening can introduce false gradients.

    if sharpness < 0.001
        info.operations{end+1} = 'Low sharpness detected - unchanged';
    end

    % ---------------------------------------------------------
    % Safety
    % ---------------------------------------------------------

    enhanced = min(max(enhanced, 0), 255);

    % ---------------------------------------------------------
    % Restore original type
    % ---------------------------------------------------------

    if strcmp(originalClass, 'double')

        out = im2double(enhanced);

    elseif strcmp(originalClass, 'single')

        out = im2single(enhanced);

    else

        out = uint8(enhanced);

    end

    info.numOperations = length(info.operations);

end