% Remaining abstract items, measured on the tuning split (PARSE test 1-100):
%   - CLAHE (abstract stage 2, "Contrast Enhancement")
%   - Adaptive HOG scale sampling: 15 instead of 10 pyramid levels per octave
% Rule fixed before running: an item is adopted only if it raises the mean
% PCK over the 5 conditions by at least 0.3 points on images 1-100.
addpath(fullfile(pwd,'tuning')); addpath(pwd);
conds = {'clean','dark','lowcontrast','noise','blur'};
S = load(fullfile('cache','tuning','quality.mat')); Q = S.Q;
tags = {'base','nlm','gamma067','deconv15'}; val = 1:100;
vote = struct('iou',0.5,'delta',1.0,'T',0.5,'r',0.25);

configs = {'current pipeline', 'CLAHE if contrast<0.46', 'CLAHE if contrast<0.60', ...
           'CLAHE on every ungated image', 'dense pyramid (15/octave)'};
P = zeros(numel(configs), 5); A = P;
for c = 1:5
    q = Q.(conds{c});
    tag = ones(205,1);
    tag(q(:,1) < 0.40) = 3; tag(q(:,4) < 0.014) = 4; tag(q(:,3) > 6) = 2;
    for k = 1:numel(configs)
        pairs = cell(1, numel(val));
        for i = 1:numel(val)
            n = val(i); t = tags{tag(n)};
            switch k
                case 2, if tag(n) == 1 && q(n,2) < 0.46, t = 'clahe'; end
                case 3, if tag(n) == 1 && q(n,2) < 0.60, t = 'clahe'; end
                case 4, if tag(n) == 1, t = 'clahe'; end
            end
            if k == 5
                pairs{i} = {'gated@i15', 'gated_flip@i15'};
            else
                pairs{i} = {t, [t '_flip']};
            end
        end
        s = struct('variants',{{}},'perImage',{pairs},'cond',conds{c},'idx',val,'vote',vote,'refine',true);
        x = tune_eval(s); P(k,c) = x.pck; A(k,c) = x.apk;
    end
end
for k = 1:numel(configs)
    fprintf('%-30s mean PCK %.2f (%+.2f) APK %.2f | %s\n', configs{k}, 100*mean(P(k,:)), ...
        100*(mean(P(k,:)) - mean(P(1,:))), 100*mean(A(k,:)), mat2str(round(100*P(k,:),1)));
end
