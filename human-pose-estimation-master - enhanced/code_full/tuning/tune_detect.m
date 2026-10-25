function tune_detect(variants, conds, idxList)
% TUNE_DETECT
% Runs baseline FMP inference on transformed PARSE test images and caches
% the top-K raw detections (mapped back to original image coordinates).
%
%   variants - cell array of variant names (see tune_transform)
%   conds    - cell array from {'clean','dark','lowcontrast','noise','blur'}
%   idxList  - indices into the 205-image PARSE test list
%
% Cache: code_full/cache/tuning/<variant>/<cond>/<idx>.mat

    codeFullDir = fileparts(fileparts(mfilename('fullpath')));
    cd(codeFullDir);
    addpath(codeFullDir, fullfile(codeFullDir, 'tuning'), fullfile(codeFullDir, 'detection'));
    globals;
    S = load(fullfile('..', 'code_basic', 'PARSE_model.mat'), 'model');
    model = S.model;
    model.thresh = min(model.thresh, -1);
    [~, test] = PARSE_data('PARSE');

    K = 500;
    mirror26 = [1 2 15:26 3:14];

    for v = 1:numel(variants)
        for c = 1:numel(conds)
            outDir = fullfile(codeFullDir, 'cache', 'tuning', variants{v}, conds{c});
            if ~exist(outDir, 'dir'), mkdir(outDir); end
            for n = idxList
                outFile = fullfile(outDir, sprintf('%03d.mat', n));
                if exist(outFile, 'file'), continue; end

                im = load_test_image(test, n, conds{c});
                % '@i15' suffix: denser HOG scale pyramid (15 levels per octave)
                [name, modelV] = parse_variant(variants{v}, model);
                t = tic;
                [imT, scale, flipped] = tune_transform(im, name);
                tPre = toc(t);
                boxes = detect_fast(imT, modelV, modelV.thresh);
                tDet = toc(t) - tPre;

                if ~isempty(boxes)
                    [~, I] = sort(boxes(:, end), 'descend');
                    boxes = boxes(I(1:min(K, numel(I))), :);
                    boxes = map_back(boxes, scale, flipped, size(imT, 2), mirror26);
                end
                boxes = single(boxes); %#ok<NASGU>
                save(outFile, 'boxes', 'tPre', 'tDet');
            end
            fprintf('[%s] %s %s done (%d images)\n', datestr(now, 'HH:MM:SS'), variants{v}, conds{c}, numel(idxList));
        end
    end
end

function [name, model] = parse_variant(name, model)
    k = strfind(name, '@i');
    if ~isempty(k)
        model.interval = str2double(name(k+2:end));
        name = name(1:k-1);
    end
end

function im = load_test_image(test, n, cond)
    if strcmp(cond, 'clean')
        im = imread(test(n).im);
    else
        im = imread(fullfile('data', 'PARSE_degraded', cond, sprintf('im%04d.jpg', 100 + n)));
    end
end

function boxes = map_back(boxes, scale, flipped, widthT, mirror26)
    numparts = floor((size(boxes, 2) - 2) / 5);
    for p = 1:numparts
        c = 5 * (p - 1);
        if flipped
            x1 = widthT + 1 - boxes(:, c + 3);
            x2 = widthT + 1 - boxes(:, c + 1);
            boxes(:, c + 1) = x1;
            boxes(:, c + 3) = x2;
        end
        boxes(:, c + (1:4)) = boxes(:, c + (1:4)) / scale;
    end
    if flipped
        % Mirrored image: the model's left parts are the person's right parts
        cols = reshape(1:5 * numparts, 5, numparts);
        cols = cols(:, mirror26);
        boxes(:, 1:5 * numparts) = boxes(:, cols(:)');
    end
end
