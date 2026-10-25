% Image-quality measures for every test image under every condition,
% used to choose gating thresholds on the validation split.
addpath(fullfile(pwd, 'tuning')); addpath(pwd);
globals;
[~, test] = PARSE_data('PARSE');
conds = {'clean','dark','lowcontrast','noise','blur'};
Q = struct();
for c = 1:numel(conds)
    M = zeros(205, 4);
    for n = 1:205
        if c == 1
            im = imread(test(n).im);
        else
            im = imread(fullfile('data','PARSE_degraded',conds{c},sprintf('im%04d.jpg',100+n)));
        end
        q = image_quality(im);
        M(n,:) = [q.brightness q.contrast q.noiseSigma q.sharpness];
    end
    Q.(conds{c}) = M;
    fprintf('%-12s bright %s | contrast %s | noise %s | sharp %s\n', conds{c}, ...
        rng5(M(:,1)), rng5(M(:,2)), rng5(M(:,3)), rng5(M(:,4)));
end
save(fullfile('cache','tuning','quality.mat'), 'Q');

function s = rng5(x)
    p = prctile(x, [0 5 50 95 100]);
    s = sprintf('%.4g/%.4g/%.4g/%.4g/%.4g', p);
end
