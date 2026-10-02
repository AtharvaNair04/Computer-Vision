function [refined_det, totalCorrections] = apply_pose_refinement(det, params)
% APPLY_POSE_REFINEMENT
% Applies anatomical pose refinement (Box 5) across an entire FMP
% detection or pose struct array.
%
% Supports:
%   - 14-point PARSE detections (e.g. after PARSE_transback)
%   - 26-point FMP detections (maps 14 joints, refines them, and updates 26-part struct)
%
% Inputs:
%   det    - struct array of detections or poses
%   params - (optional) refinement parameters
%
% Output:
%   refined_det      - updated detection struct array
%   totalCorrections - total number of joint corrections performed

    if nargin < 2
        params = struct();
    end

    refined_det = det;
    totalCorrections = 0;

    I = [1  2  3  4  5  6  7  8  9  10 11 12 13 14];
    J = [14 12 10 22 24 26 7  5  3  15 17 19 2  1];

    for n = 1:numel(det)
        for i = 1:numel(det(n).obj)
            if isempty(det(n).obj(i).point)
                continue;
            end

            pts = det(n).obj(i).point;

            if size(pts, 1) == 14 && size(pts, 2) == 2
                % Direct 14-point refinement
                [ptsRefined, info] = refine_pose(pts, params);
                refined_det(n).obj(i).point = ptsRefined;
                totalCorrections = totalCorrections + info.numCorrections;

            elseif size(pts, 1) == 26 && size(pts, 2) == 2
                % 26-point FMP representation:
                % Extract 14 joints
                pts14 = pts(J, :);
                [pts14Refined, info] = refine_pose(pts14, params);
                totalCorrections = totalCorrections + info.numCorrections;

                % Compute delta offsets on the 14 primary joints
                delta14 = pts14Refined - pts14;
                ptsRefined = pts;
                ptsRefined(J, :) = pts14Refined;

                % Update intermediate midpoint parts (e.g. part 4 between 3 and 5, part 16 between 15 and 17, etc.)
                % For each joint k with a parent in FMP tree, smooth intermediate parts
                for k = 1:14
                    idxJoint = J(k);
                    % Propagate 50% shift to adjacent intermediate part if applicable
                    if idxJoint + 1 <= 26 && ~ismember(idxJoint + 1, J)
                        ptsRefined(idxJoint + 1, :) = ptsRefined(idxJoint + 1, :) + 0.5 * delta14(k, :);
                    end
                    if idxJoint - 1 >= 1 && ~ismember(idxJoint - 1, J)
                        ptsRefined(idxJoint - 1, :) = ptsRefined(idxJoint - 1, :) + 0.5 * delta14(k, :);
                    end
                end

                refined_det(n).obj(i).point = ptsRefined;
            end
        end
    end
end
