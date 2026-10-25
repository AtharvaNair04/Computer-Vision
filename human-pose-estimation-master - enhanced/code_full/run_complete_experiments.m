function results = run_complete_experiments(idx)
% RUN_COMPLETE_EXPERIMENTS
% Entry point kept for compatibility; the evaluation now lives in
% run_final_evaluation.m (the previous 25-image version is in git history).
%
%   results = run_complete_experiments()         % held-out PARSE test 101-205
%   results = run_complete_experiments(1:205)    % full PARSE test set
%
% For BUFFY run run_buffy_experiments().

    if nargin < 1 || isempty(idx)
        idx = 101:205;
    end
    results = run_final_evaluation(idx);
end
