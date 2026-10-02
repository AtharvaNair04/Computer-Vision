function plot_parse_skeleton(im, points, options)
% PLOT_PARSE_SKELETON
% Publication-quality rendering of the 14 PARSE joints and kinematic skeleton.
%
% Inputs:
%   im      - RGB or grayscale background image
%   points  - 14 x 2 matrix of keypoints [x, y]
%   options - (optional) struct with display settings:
%               .lineWidth   (default: 3.5)
%               .markerSize  (default: 7)
%               .title       (default: '')
%               .showLabels  (default: false)

    if nargin < 3
        options = struct();
    end
    if ~isfield(options, 'lineWidth'),  options.lineWidth = 3.5; end
    if ~isfield(options, 'markerSize'), options.markerSize = 6;  end
    if ~isfield(options, 'title'),      options.title = '';      end
    if ~isfield(options, 'showLabels'), options.showLabels = false; end

    if ~isempty(im)
        imagesc(im);
        axis image; axis off;
        hold on;
    else
        hold on;
    end

    if isempty(points) || size(points, 1) ~= 14
        return;
    end

    % 14 PARSE Joints:
    % 1: R Ankle, 2: R Knee, 3: R Hip, 4: L Hip, 5: L Knee, 6: L Ankle
    % 7: R Wrist, 8: R Elbow, 9: R Shoulder, 10: L Shoulder, 11: L Elbow, 12: L Wrist
    % 13: Neck, 14: Head

    % Kinematic bones [joint1, joint2, R G B]
    % Right side = Red/Orange, Left side = Cyan/Blue, Center = Yellow/Green
    limbs = {
        % Right leg
        [1, 2], [0.95, 0.25, 0.15];   % R Ankle - R Knee
        [2, 3], [0.95, 0.40, 0.10];   % R Knee - R Hip
        % Left leg
        [6, 5], [0.15, 0.65, 0.95];   % L Ankle - L Knee
        [5, 4], [0.20, 0.50, 0.95];   % L Knee - L Hip
        % Hips / Pelvis
        [3, 4], [0.95, 0.85, 0.10];   % R Hip - L Hip
        % Torso
        [3, 9],  [0.85, 0.85, 0.20];  % R Hip - R Shoulder
        [4, 10], [0.85, 0.85, 0.20];  % L Hip - L Shoulder
        [9, 10], [0.95, 0.85, 0.10];  % R Shoulder - L Shoulder
        % Right arm
        [9, 8], [0.95, 0.50, 0.20];   % R Shoulder - R Elbow
        [8, 7], [0.95, 0.25, 0.35];   % R Elbow - R Wrist
        % Left arm
        [10, 11], [0.20, 0.75, 0.85]; % L Shoulder - L Elbow
        [11, 12], [0.20, 0.90, 0.70]; % L Elbow - L Wrist
        % Neck and Head
        [9, 13],  [0.40, 0.95, 0.35]; % R Shoulder - Neck
        [10, 13], [0.40, 0.95, 0.35]; % L Shoulder - Neck
        [13, 14], [0.95, 0.95, 0.95]; % Neck - Head
    };

    % Draw limb lines
    for i = 1:size(limbs, 1)
        pair = limbs{i, 1};
        col  = limbs{i, 2};
        p1 = points(pair(1), :);
        p2 = points(pair(2), :);
        line([p1(1), p2(1)], [p1(2), p2(2)], ...
             'Color', col, 'LineWidth', options.lineWidth);
    end

    % Draw joint markers
    for j = 1:14
        pj = points(j, :);
        if ismember(j, [1, 2, 7, 8])
            jColor = [0.95, 0.2, 0.2]; % Right side joint: Red
        elseif ismember(j, [5, 6, 11, 12])
            jColor = [0.2, 0.6, 0.95]; % Left side joint: Blue
        else
            jColor = [1.0, 0.9, 0.1]; % Center / Torso: Gold
        end

        plot(pj(1), pj(2), 'o', ...
             'MarkerSize', options.markerSize, ...
             'MarkerFaceColor', jColor, ...
             'MarkerEdgeColor', [0 0 0], ...
             'LineWidth', 1.2);

        if options.showLabels
            text(pj(1) + 3, pj(2) - 3, sprintf('%d', j), ...
                 'Color', 'w', 'FontSize', 8, 'FontWeight', 'bold');
        end
    end

    if ~isempty(options.title)
        title(options.title, 'FontSize', 11, 'FontWeight', 'bold', 'Color', [0 0 0]);
    end
end
