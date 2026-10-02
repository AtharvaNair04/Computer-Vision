function [out, info] = adaptive_preprocess_v2(im)
% ADAPTIVE_PREPROCESS_V2
%
% Adaptive image preprocessing based on:
%   - brightness
%   - contrast
%   - noise
%   - sharpness
%
% Handles uint8 images and double images in either:
%   0-1 range
%   0-255 range

% ------------------------------------------------------------
% Preserve original class
% ------------------------------------------------------------
originalClass = class(im);

% ------------------------------------------------------------
% Convert input safely to uint8
% ------------------------------------------------------------
if isa(im,'uint8')

    workingImage = im;

elseif isa(im,'double') || isa(im,'single')

    maxValue = max(im(:));

    if maxValue <= 1

        % Double/single image already in 0-1 range
        workingImage = im2uint8(im);

    else

        % Double/single image is in 0-255 range
        workingImage = uint8(min(max(im,0),255));

    end

else

    workingImage = im2uint8(im);

end

% ------------------------------------------------------------
% Convert to grayscale for image statistics
% ------------------------------------------------------------
if size(workingImage,3) == 3
    gray = rgb2gray(workingImage);
else
    gray = workingImage;
end

grayDouble = im2double(gray);

% ------------------------------------------------------------
% Calculate image statistics
% ------------------------------------------------------------

brightness = mean(grayDouble(:));

contrast = std(grayDouble(:));

% Estimate noise using high-frequency residual
smoothImage = imgaussfilt(grayDouble,1);

residual = grayDouble - smoothImage;

noiseLevel = std(residual(:));

% Estimate sharpness using Laplacian
laplacianKernel = [0 1 0;
                   1 -4 1;
                   0 1 0];

laplacianResponse = imfilter( ...
    grayDouble, ...
    laplacianKernel, ...
    'replicate');

sharpness = var(laplacianResponse(:));

% ------------------------------------------------------------
% Store statistics
% ------------------------------------------------------------

info.brightness = brightness;
info.contrast = contrast;
info.noise = noiseLevel;
info.sharpness = sharpness;

info.operations = {};

% ------------------------------------------------------------
% Start with original image
% ------------------------------------------------------------

enhanced = workingImage;

% ------------------------------------------------------------
% Brightness correction
% ------------------------------------------------------------

if brightness < 0.25

    % Dark image
    gamma = 0.85;

    enhanced = imadjust( ...
        enhanced, ...
        [], ...
        [], ...
        gamma);

    info.operations{end+1} = 'Mild gamma correction';

elseif brightness > 0.80

    % Bright image
    gamma = 1.10;

    enhanced = imadjust( ...
        enhanced, ...
        [], ...
        [], ...
        gamma);

    info.operations{end+1} = 'Mild brightness correction';

end

% ------------------------------------------------------------
% Contrast enhancement
% ------------------------------------------------------------

if contrast < 0.12

    if size(enhanced,3) == 3

        lab = rgb2lab(enhanced);

        L = lab(:,:,1) / 100;

        low = min(L(:));
        high = max(L(:));

        if high > low

            L = (L-low)/(high-low);

        end

        lab(:,:,1) = L * 100;

        enhanced = lab2rgb( ...
            lab, ...
            'OutputType','uint8');

    else

        low = double(min(enhanced(:)));
        high = double(max(enhanced(:)));

        if high > low

            enhanced = uint8( ...
                255 * ...
                (double(enhanced)-low) / ...
                (high-low));

        end

    end

    info.operations{end+1} = ...
        'Gentle contrast enhancement';

end

% ------------------------------------------------------------
% Noise detection
% ------------------------------------------------------------

if noiseLevel > 0.08

    % Do not aggressively denoise because this can
    % remove pose boundaries needed by HOG.

    info.operations{end+1} = ...
        'High noise detected - unchanged';

end

% ------------------------------------------------------------
% Sharpness detection
% ------------------------------------------------------------

if sharpness < 0.001

    % Do not aggressively sharpen because this can
    % introduce artificial edges.

    info.operations{end+1} = ...
        'Low sharpness detected - unchanged';

end

% ------------------------------------------------------------
% Safety clamp
% ------------------------------------------------------------

enhanced = uint8( ...
    min(max(enhanced,0),255));

% ------------------------------------------------------------
% Restore original output class
% ------------------------------------------------------------

if strcmp(originalClass,'double')

    % Return double in 0-255 because the detector receives
    % double images in this format.

    out = double(enhanced);

elseif strcmp(originalClass,'single')

    out = single(enhanced);

else

    out = uint8(enhanced);

end

% ------------------------------------------------------------
% Number of operations
% ------------------------------------------------------------

info.numOperations = length(info.operations);

end