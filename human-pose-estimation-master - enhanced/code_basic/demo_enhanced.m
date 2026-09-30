addpath visualization;
addpath mex_pc;

compile;

% Load model
load('PARSE_model.mat');

% Images
imlist = dir('images/*.jpg');

for i = 1:length(imlist)

    % Load image
    im = imread(fullfile('images', imlist(i).name));

    % -----------------------------------------
    % OUR NOVELTY: ADAPTIVE PREPROCESSING
    % -----------------------------------------
    [im_enhanced, info] = adaptive_preprocess(im);

    % -----------------------------------------
    % ORIGINAL FMP
    % -----------------------------------------
    tic;
    boxes = detect_fast(im_enhanced, model, min(model.thresh,-1));
    dettime = toc;

    boxes = nms(boxes, .1);

    % Convert detection to pose
    det = box2obj(boxes);

    % -----------------------------------------
    % DISPLAY
    % -----------------------------------------
    figure;

    imagesc(im_enhanced);
    axis image;
    axis off;

    colorset = {'g','g','y','m','m','m','m','y','y','y', ...
        'r','r','r','r','y','c','c','c','c','y', ...
        'y','y','b','b','b','b'};

    showboxes(im_enhanced, det(1), colorset);

    title(sprintf('Adaptive FMP - %s', imlist(i).name));

    fprintf('\nImage: %s\n', imlist(i).name);
    fprintf('Preprocessing operations: %d\n', info.numOperations);
    fprintf('Detection time: %.2f seconds\n', dettime);

    disp('Press any key to continue...');
    pause;

    close all;
end