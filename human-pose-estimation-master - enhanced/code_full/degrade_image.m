function out = degrade_image(im, condition)
% DEGRADE_IMAGE
% Creates controlled image degradations for robustness testing.
%
% condition:
%   'dark'
%   'lowcontrast'
%   'noise'
%   'blur'

% Make sure image is uint8
if ~isa(im, 'uint8')
    im = im2uint8(im);
end

switch lower(condition)

    case 'dark'
        % Darken the image using gamma correction
        out = imadjust(im, [], [], 1.5);

    case 'lowcontrast'
        % Reduce global contrast
        if size(im, 3) == 3
            lab = rgb2lab(im);

            L = lab(:,:,1) / 100;

            % Compress luminance around the middle
            L = 0.5 + 0.45 * (L - 0.5);

            lab(:,:,1) = L * 100;

            out = lab2rgb(lab, 'OutputType', 'uint8');
        else
            x = im2double(im);
            x = 0.5 + 0.45 * (x - 0.5);
            out = im2uint8(x);
        end

    case 'noise'
        % Moderate Gaussian noise
        out = imnoise(im, 'gaussian', 0, 0.01);

    case 'blur'
        % Moderate Gaussian blur
        out = imgaussfilt(im, 1.5);

    otherwise
        error('Unknown degradation condition: %s', condition);
end

% Safety
out = uint8(min(max(out, 0), 255));
end