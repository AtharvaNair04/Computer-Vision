function pos = make_adaptive_training_set()
% MAKE_ADAPTIVE_TRAINING_SET
% Build a PARSE training set for adaptive-FMP fine-tuning.
%
% Uses only PARSE training frames 1:100 (never the 101:305 test set).
% For each training image we keep:
%   1 clean image
%   1 dark version
%   1 low-contrast version
%   1 noisy version
%   1 blurred version
%
% Ground-truth boxes/points are unchanged by these image-only degradations.
%
% Output:
%   pos - 500 positive examples (100 clean + 4*100 degraded)
%
% The generated images are cached under:
%   code_full/cache/adaptive_train_degraded/<condition>/

    globals;

    % Base PARSE training annotations. This also creates the standard
    % PARSE cache if it does not exist.
    [posBase, ~] = PARSE_data('PARSE');

    % PARSE_data returns rotations/flips as well. We deliberately use
    % only the first 100 original training frames for the degraded
    % augmentation so that the training run remains practical.
    if numel(posBase) < 100
        error('PARSE training data contains fewer than 100 original examples.');
    end
    posClean = posBase(1:100);

    conditions = {'dark','lowcontrast','noise','blur'};
    cacheRoot = fullfile(cachedir, 'adaptive_train_degraded');

    for c = 1:numel(conditions)
        outDir = fullfile(cacheRoot, conditions{c});
        if ~exist(outDir, 'dir')
            mkdir(outDir);
        end
    end

    % Make the stochastic noise augmentation reproducible.
    rng(0, 'twister');

    pos = posClean;
    next = numel(pos) + 1;

    for c = 1:numel(conditions)
        condition = conditions{c};
        outDir = fullfile(cacheRoot, condition);

        fprintf('\nGenerating/using %s training images...\n', upper(condition));

        for n = 1:100
            src = posClean(n).im;
            [~, name, ext] = fileparts(src);
            outFile = fullfile(outDir, [name ext]);

            if ~exist(outFile, 'file')
                im = imread(src);
                degraded = degrade_image(im, condition);
                imwrite(degraded, outFile, 'jpg', 'Quality', 95);
            end

            ex = posClean(n);
            ex.im = outFile;
            pos(next) = ex; %#ok<AGROW>
            next = next + 1;
        end
    end

    fprintf('\nAdaptive training set ready: %d positives\n', numel(pos));
    fprintf('  Clean:       100\n');
    fprintf('  Dark:        100\n');
    fprintf('  Low contrast:100\n');
    fprintf('  Noise:       100\n');
    fprintf('  Blur:        100\n');
end
