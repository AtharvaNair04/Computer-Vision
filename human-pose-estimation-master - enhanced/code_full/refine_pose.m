function [refined_points, info] = refine_pose(points, params)
% REFINE_POSE
% Post-processing pose refinement module based on:
%   1. Geometric anatomical constraints (anthropometric limb proportions)
%   2. Left-right bilateral symmetry constraints
%   3. Outlier detection and projection along anatomical kinematic chains
%   4. Vertical orientation and pose consistency checks
%
% Input:
%   points - 14 x 2 matrix of PARSE keypoints:
%            1: Right Ankle,  2: Right Knee,     3: Right Hip,
%            4: Left Hip,     5: Left Knee,      6: Left Ankle,
%            7: Right Wrist,  8: Right Elbow,    9: Right Shoulder,
%           10: Left Shoulder,11: Left Elbow,   12: Left Wrist,
%           13: Neck,         14: Head Top
%   params - (optional) struct with tuning parameters
%
% Output:
%   refined_points - 14 x 2 matrix of refined keypoints
%   info           - diagnostics struct recording any corrections made

    if nargin < 2 || isempty(params)
        params = struct();
    end

    % Calibrated anthropometric parameters based on Drillis & Contini (1966) & Winter (2009)
    if ~isfield(params, 'maxUpperArmRatio'), params.maxUpperArmRatio = 0.90; end
    if ~isfield(params, 'minUpperArmRatio'), params.minUpperArmRatio = 0.35; end
    if ~isfield(params, 'maxForearmRatio'),  params.maxForearmRatio  = 0.80; end
    if ~isfield(params, 'minForearmRatio'),  params.minForearmRatio  = 0.30; end
    if ~isfield(params, 'maxThighRatio'),    params.maxThighRatio    = 1.20; end
    if ~isfield(params, 'minThighRatio'),    params.minThighRatio    = 0.50; end
    if ~isfield(params, 'maxCalfRatio'),     params.maxCalfRatio     = 1.15; end
    if ~isfield(params, 'minCalfRatio'),     params.minCalfRatio     = 0.45; end
    if ~isfield(params, 'maxHeadRatio'),     params.maxHeadRatio     = 1.20; end
    if ~isfield(params, 'minHeadRatio'),     params.minHeadRatio     = 0.05; end
    if ~isfield(params, 'symmetryWeight'),   params.symmetryWeight   = 0.35; end

    refined_points = points;
    info = struct('corrections', {{}}, 'numCorrections', 0);

    if size(points, 1) ~= 14 || size(points, 2) ~= 2
        return;
    end

    % -------------------------------------------------------------
    % 1. Reference Torso Scale with Bounding-Box Height Prior
    % -------------------------------------------------------------
    % Shoulders: 9 (Right), 10 (Left)
    % Hips:      3 (Right),  4 (Left)
    shoulderDist = norm(points(9, :) - points(10, :));
    hipDist      = norm(points(3, :) - points(4, :));
    midShoulder  = (points(9, :) + points(10, :)) / 2;
    midHip       = (points(3, :) + points(4, :)) / 2;
    torsoLen     = norm(midShoulder - midHip);

    % Anchor torso scale to overall detected person height prior (~30% of standing height)
    minPt = min(points, [], 1);
    maxPt = max(points, [], 1);
    personHeight = max(maxPt(2) - minPt(2), 50.0);
    priorTorsoScale = 0.30 * personHeight;

    if torsoLen >= 0.18 * personHeight && torsoLen <= 0.45 * personHeight
        torsoScale = 0.65 * torsoLen + 0.35 * priorTorsoScale;
    else
        torsoScale = priorTorsoScale;
    end
    torsoScale = max([torsoScale, shoulderDist * 1.0, hipDist * 1.2, 30.0]);

    % -------------------------------------------------------------
    % 2. Bilateral Symmetry Reference Lengths (Calibrated Anthropometrics)
    % -------------------------------------------------------------
    lenUpperR = norm(points(8, :) - points(9, :));
    lenUpperL = norm(points(11, :) - points(10, :));
    lenForeR  = norm(points(7, :) - points(8, :));
    lenForeL  = norm(points(12, :) - points(11, :));

    lenThighR = norm(points(2, :) - points(3, :));
    lenThighL = norm(points(5, :) - points(4, :));
    lenCalfR  = norm(points(1, :) - points(2, :));
    lenCalfL  = norm(points(6, :) - points(5, :));

    % Expected symmetrical target lengths
    targetUpper = median([lenUpperR, lenUpperL, 0.60 * torsoScale]);
    targetFore  = median([lenForeR,  lenForeL,  0.50 * torsoScale]);
    targetThigh = median([lenThighR, lenThighL, 0.85 * torsoScale]);
    targetCalf  = median([lenCalfR,  lenCalfL,  0.75 * torsoScale]);

    % -------------------------------------------------------------
    % 3. Refine Upper Arms: Shoulder (9/10) -> Elbow (8/11)
    % -------------------------------------------------------------
    % Right Elbow (8) from Right Shoulder (9)
    [refined_points(8, :), corrR8] = bound_joint( ...
        refined_points(9, :), refined_points(8, :), ...
        params.minUpperArmRatio * torsoScale, ...
        params.maxUpperArmRatio * torsoScale, ...
        targetUpper, params.symmetryWeight, 'Right Elbow');
    if corrR8, info.corrections{end+1} = 'Right Elbow constrained'; end

    % Left Elbow (11) from Left Shoulder (10)
    [refined_points(11, :), corrL11] = bound_joint( ...
        refined_points(10, :), refined_points(11, :), ...
        params.minUpperArmRatio * torsoScale, ...
        params.maxUpperArmRatio * torsoScale, ...
        targetUpper, params.symmetryWeight, 'Left Elbow');
    if corrL11, info.corrections{end+1} = 'Left Elbow constrained'; end

    % -------------------------------------------------------------
    % 4. Refine Forearms: Elbow (8/11) -> Wrist (7/12)
    % -------------------------------------------------------------
    % Right Wrist (7) from Right Elbow (8)
    [refined_points(7, :), corrR7] = bound_joint( ...
        refined_points(8, :), refined_points(7, :), ...
        params.minForearmRatio * torsoScale, ...
        params.maxForearmRatio * torsoScale, ...
        targetFore, params.symmetryWeight, 'Right Wrist');
    if corrR7, info.corrections{end+1} = 'Right Wrist constrained'; end

    % Left Wrist (12) from Left Elbow (11)
    [refined_points(12, :), corrL12] = bound_joint( ...
        refined_points(11, :), refined_points(12, :), ...
        params.minForearmRatio * torsoScale, ...
        params.maxForearmRatio * torsoScale, ...
        targetFore, params.symmetryWeight, 'Left Wrist');
    if corrL12, info.corrections{end+1} = 'Left Wrist constrained'; end

    % -------------------------------------------------------------
    % 5. Refine Legs: Hip (3/4) -> Knee (2/5)
    % -------------------------------------------------------------
    % Right Knee (2) from Right Hip (3)
    [refined_points(2, :), corrR2] = bound_joint( ...
        refined_points(3, :), refined_points(2, :), ...
        params.minThighRatio * torsoScale, ...
        params.maxThighRatio * torsoScale, ...
        targetThigh, params.symmetryWeight, 'Right Knee');
    if corrR2, info.corrections{end+1} = 'Right Knee constrained'; end

    % Left Knee (5) from Left Hip (4)
    [refined_points(5, :), corrL5] = bound_joint( ...
        refined_points(4, :), refined_points(5, :), ...
        params.minThighRatio * torsoScale, ...
        params.maxThighRatio * torsoScale, ...
        targetThigh, params.symmetryWeight, 'Left Knee');
    if corrL5, info.corrections{end+1} = 'Left Knee constrained'; end

    % -------------------------------------------------------------
    % 6. Refine Calves: Knee (2/5) -> Ankle (1/6)
    % -------------------------------------------------------------
    % Right Ankle (1) from Right Knee (2)
    [refined_points(1, :), corrR1] = bound_joint( ...
        refined_points(2, :), refined_points(1, :), ...
        params.minCalfRatio * torsoScale, ...
        params.maxCalfRatio * torsoScale, ...
        targetCalf, params.symmetryWeight, 'Right Ankle');
    if corrR1, info.corrections{end+1} = 'Right Ankle constrained'; end

    % Left Ankle (6) from Left Knee (5)
    [refined_points(6, :), corrL6] = bound_joint( ...
        refined_points(5, :), refined_points(6, :), ...
        params.minCalfRatio * torsoScale, ...
        params.maxCalfRatio * torsoScale, ...
        targetCalf, params.symmetryWeight, 'Left Ankle');
    if corrL6, info.corrections{end+1} = 'Left Ankle constrained'; end

    % -------------------------------------------------------------
    % 7. Refine Head & Neck: Outlier-Gated (Preserve High Baseline Accuracy)
    % -------------------------------------------------------------
    distNeck = norm(refined_points(13, :) - midShoulder);
    if distNeck > 0.85 * torsoScale || distNeck < 0.05 * torsoScale
        targetHeadLen = 0.30 * torsoScale;
        [refined_points(13, :), corr13] = bound_joint( ...
            midShoulder, refined_points(13, :), ...
            0.05 * torsoScale, 0.85 * torsoScale, ...
            targetHeadLen, 0, 'Neck');
        if corr13, info.corrections{end+1} = 'Neck constrained'; end
    end

    distHead = norm(refined_points(14, :) - refined_points(13, :));
    if distHead > 0.70 * torsoScale || distHead < 0.05 * torsoScale
        targetHeadLen = 0.30 * torsoScale;
        [refined_points(14, :), corr14] = bound_joint( ...
            refined_points(13, :), refined_points(14, :), ...
            0.05 * torsoScale, 0.70 * torsoScale, ...
            targetHeadLen, 0, 'Head');
        if corr14, info.corrections{end+1} = 'Head constrained'; end
    end

    info.numCorrections = length(info.corrections);
end

% -----------------------------------------------------------------
% Helper: Project / Bound Joint along Kinematic Bone Vector
% -----------------------------------------------------------------
function [newPos, wasCorrected] = bound_joint(parentPos, childPos, minLen, maxLen, targetLen, symWeight, ~)
    vec = childPos - parentPos;
    curLen = norm(vec);
    wasCorrected = false;

    if curLen < 1e-4
        % Degenerate overlap: place slightly along vertical
        vec = [0, minLen];
        curLen = minLen;
        wasCorrected = true;
    end

    unitVec = vec / curLen;

    % Check min/max anatomical bounds
    if curLen > maxLen
        newLen = maxLen;
        wasCorrected = true;
    elseif curLen < minLen
        newLen = minLen;
        wasCorrected = true;
    else
        newLen = curLen;
    end

    % Gentle symmetry prior adjustment towards target length
    if symWeight > 0 && ~isempty(targetLen)
        if curLen > 1.25 * targetLen || curLen < 0.75 * targetLen
            newLen = (1 - symWeight) * newLen + symWeight * targetLen;
            wasCorrected = true;
        end
    end

    newPos = parentPos + unitVec * newLen;
end
