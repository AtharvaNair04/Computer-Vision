function generate_degraded_parse()

% Generate controlled degraded versions of all PARSE test images.
%
% Test images:
%   im0101.jpg ... im0305.jpg
%
% Conditions:
%   dark
%   lowcontrast
%   noise
%   blur

baseDir = fileparts(mfilename('fullpath'));

inputDir = fullfile(baseDir, 'data', 'PARSE');

outputDir = fullfile(baseDir, 'data', 'PARSE_degraded');

conditions = {'dark', 'lowcontrast', 'noise', 'blur'};

% Create output folders
for c = 1:length(conditions)
    folder = fullfile(outputDir, conditions{c});

    if ~exist(folder, 'dir')
        mkdir(folder);
    end
end

fprintf('Generating degraded PARSE images...\n');

for i = 101:305

    filename = sprintf('im%04d.jpg', i);
    inputFile = fullfile(inputDir, filename);

    if ~exist(inputFile, 'file')
        warning('Missing image: %s', inputFile);
        continue;
    end

    im = imread(inputFile);

    for c = 1:length(conditions)

        condition = conditions{c};

        degraded = degrade_image(im, condition);

        outputFile = fullfile( ...
            outputDir, ...
            condition, ...
            filename);

        imwrite(degraded, outputFile);
    end

    if mod(i - 100, 10) == 0
        fprintf('Processed %d / 205 images\n', i - 100);
    end
end

fprintf('\nFinished generating degraded dataset.\n');
fprintf('Output folder:\n%s\n', outputDir);

end