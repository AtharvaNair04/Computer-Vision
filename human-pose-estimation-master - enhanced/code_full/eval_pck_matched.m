function pck = eval_pck_matched(det, test, thresh)
% EVAL_PCK_MATCHED
%
% Computes PCK for PARSE using the FMP 26-point detections
% converted to the original 14 PARSE joints.
%
% This follows the same 26 -> 14 mapping used by:
%     PARSE_transback.m
%
% Inputs:
%   det    - detection results from box2obj()
%   test   - PARSE ground truth
%   thresh - normalized PCK threshold
%
% Output:
%   pck    - 1 x 14 PCK values

    numImages = length(test);
    numJoints = 14;

    % -------------------------------------------------------------
    % Same transformation used by PARSE_transback.m
    % -------------------------------------------------------------

    I = [1  2  3  4  5  6  7  8  9  10 11 12 13 14];

    J = [14 12 10 22 24 26 7  5  3  15 17 19 2  1];

    A = ones(1,14);

    Transback = full(sparse(I, J, A, 14, 26));


    % -------------------------------------------------------------
    % Counters
    % -------------------------------------------------------------

    correct = zeros(1, numJoints);

    total = zeros(1, numJoints);


    % -------------------------------------------------------------
    % Process every test image
    % -------------------------------------------------------------

    for i = 1:numImages

        % Ground truth
        gtObj = test(i).obj(1);

        gtPoints = gtObj.point;


        % ---------------------------------------------------------
        % Ground truth should be 14 x 2
        % ---------------------------------------------------------

        if size(gtPoints,1) ~= 14 || size(gtPoints,2) ~= 2

            warning( ...
                'Image %d: unexpected GT size [%d %d]', ...
                i, ...
                size(gtPoints,1), ...
                size(gtPoints,2));

            continue;

        end


        % ---------------------------------------------------------
        % No detection
        % ---------------------------------------------------------

        if isempty(det(i).obj)

            total = total + 1;

            continue;

        end


        % ---------------------------------------------------------
        % Select the detection with the highest detection score
        % ---------------------------------------------------------

        bestIdx = 1;
        bestScore = -inf;

        for d = 1:length(det(i).obj)

            if isfield(det(i).obj(d), 'score')

                currentScore = det(i).obj(d).score;

            else

                currentScore = 0;

            end


            if currentScore > bestScore

                bestScore = currentScore;
                bestIdx = d;

            end

        end


        detectedPoints = det(i).obj(bestIdx).point;


        % ---------------------------------------------------------
        % Convert FMP 26 points -> PARSE 14 joints
        % ---------------------------------------------------------

        if size(detectedPoints,1) ~= 26 || ...
           size(detectedPoints,2) ~= 2

            warning( ...
                'Image %d: unexpected detection size [%d %d]', ...
                i, ...
                size(detectedPoints,1), ...
                size(detectedPoints,2));

            total = total + 1;

            continue;

        end


        detectedPoints = Transback * detectedPoints;


        % ---------------------------------------------------------
        % Calculate torso/scale normalization
        %
        % Use the same PARSE joint representation after
        % conversion.
        % ---------------------------------------------------------

        shoulder1 = gtPoints(3,:);
        shoulder2 = gtPoints(4,:);

        scale = sqrt(sum( ...
            (shoulder1 - shoulder2).^2));


        if scale <= 0

            continue;

        end


        % ---------------------------------------------------------
        % Calculate Euclidean distance for each joint
        % ---------------------------------------------------------

        distances = sqrt(sum( ...
            (detectedPoints - gtPoints).^2, ...
            2));


        % Normalize by scale
        normalizedDistance = distances ./ scale;


        % ---------------------------------------------------------
        % Count correct joints
        % ---------------------------------------------------------

        correct = correct + ...
            (normalizedDistance <= thresh)';


        total = total + 1;

    end


    % -------------------------------------------------------------
    % Final PCK
    % -------------------------------------------------------------

    pck = zeros(1, numJoints);

    valid = total > 0;

    pck(valid) = ...
        correct(valid) ./ total(valid);

end