function [det,pose] = combined_testmodel(name,model,test,suffix)
% combined_testmodel
%
% Exact counterpart of testmodel.m using
% combined_detect_fast() instead of detect_fast().
%
% Evaluation protocol is unchanged.

globals;

if nargin < 4
    suffix = [];
end

try

    load([cachedir name '_detections' suffix]);

catch

    det = struct('obj',cell(1,length(test)));
    pose = struct('obj',cell(1,length(test)));

    for i = 1:length(test)

        fprintf([name ': testing: %d/%d\n'],i,length(test));

        im = imread(test(i).im);

        % Combined detector
        boxes = combined_detect_fast( ...
            im, ...
            model, ...
            model.thresh);

        N = numel(test(i).obj);

        pose(i).obj = struct( ...
            'box',cell(1,N), ...
            'point',cell(1,N), ...
            'mixture',cell(1,N), ...
            'score',cell(1,N));

        for n = 1:N

            box = bestoverlap( ...
                boxes, ...
                test(i).obj(n), ...
                0.3);

            if isempty(box)
                continue;
            end

            pose(i).obj(n) = box2obj(box);

        end

        boxes = nms(boxes,0.3);

        det(i).obj = box2obj(boxes);

    end

    save( ...
        [cachedir name '_detections' suffix], ...
        'det', ...
        'pose', ...
        'model');

end

end