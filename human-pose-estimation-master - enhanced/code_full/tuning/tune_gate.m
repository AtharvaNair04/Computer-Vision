% Choose image-quality gating thresholds on the validation split (idx 1:100).
% Per-image PCK is independent across images, so each (condition, transform)
% is scored once and every gating policy is evaluated by table lookup.
addpath(fullfile(pwd,'tuning')); addpath(pwd);
conds = {'clean','dark','lowcontrast','noise','blur'};
tags  = {'base','nlm','gamma067','deconv15'};
vote  = struct('iou',0.5,'delta',1.0,'T',0.5,'r',0.25);
S = load(fullfile('cache','tuning','quality.mat')); Q = S.Q;

tblFile = fullfile('cache','tuning','gate_tables_v2.mat');
if exist(tblFile, 'file')
    load(tblFile, 'PI');
else
    PI = struct();
    for c = 1:5
        for t = 1:4
            for pool = 0:1
                v = {tags{t}, [tags{t} '_flip']};
                if pool && t > 1, v = [{'base','base_flip'}, v]; end
                s = struct('variants',{v},'cond',conds{c},'idx',1:205,'vote',vote,'refine',true);
                r = tune_eval(s);
                PI.(conds{c})(t, :, pool+1) = r.pckPerImage;
            end
        end
    end
    save(tblFile, 'PI');
end

val = 1:100;
best = -inf;
for pool = 1:2
 for nT = [5 6 7 8 9]
  for bT = [0 0.010 0.012 0.014 0.016 0.018 0.020]
   for dT = [0 0.25 0.30 0.35 0.40 0.45 0.50 1]
     P = zeros(1,5);
     for c = 1:5
       q = Q.(conds{c});
       tag = ones(205,1);
       tag(q(:,1) < dT) = 3;
       tag(q(:,4) < bT) = 4;
       tag(q(:,3) > nT) = 2;
       pi = PI.(conds{c})(:, :, pool);
       P(c) = mean(pi(sub2ind(size(pi), tag(val)', val)));
     end
     if mean(P) > best
       best = mean(P);
       fprintf('pool %d nT %g bT %.3f dT %.2f | mean PCK %.2f | %s\n', pool-1, nT, bT, dT, 100*mean(P), mat2str(round(100*P,1)));
     end
   end
  end
 end
end
% reference rows
for t = 1:4
  fprintf('always %-9s: %s\n', tags{t}, mat2str(round(100*cellfun(@(c) mean(PI.(c)(t,val,1)), conds),1)));
end
