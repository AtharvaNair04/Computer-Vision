% Per-joint PCK by stage on the tuning split (1:100), gated policy.
addpath(fullfile(pwd,'tuning')); addpath(pwd);
conds = {'clean','dark','lowcontrast','noise','blur'};
S = load(fullfile('cache','tuning','quality.mat')); Q = S.Q;
tags = {'base','nlm','gamma067','deconv15'}; val = 1:100;
vote = struct('iou',0.7,'delta',0.25,'T',0.1);
J = {'Head','Shou','Elbo','Wris','Hip','Knee','Ankl'};
for c = 1:5
  q = Q.(conds{c}); tag = ones(205,1);
  tag(q(:,1) < 0.40) = 3; tag(q(:,4) < 0.014) = 4; tag(q(:,3) > 6) = 2;
  single = arrayfun(@(n) tags(tag(n)), val, 'UniformOutput', false);
  pair   = arrayfun(@(n) {tags{tag(n)}, [tags{tag(n)} '_flip']}, val, 'UniformOutput', false);
  rows = {
    'base',        struct('variants',{{'base'}},'cond',conds{c},'idx',val)
    'pre',         struct('variants',{{}},'perImage',{single},'cond',conds{c},'idx',val)
    'pre+flip',    struct('variants',{{}},'perImage',{pair},'cond',conds{c},'idx',val)
    'pre+flip+vote', struct('variants',{{}},'perImage',{pair},'cond',conds{c},'idx',val,'vote',vote)
    'full',        struct('variants',{{}},'perImage',{pair},'cond',conds{c},'idx',val,'vote',vote,'refine',true)
  };
  fprintf('=== %s ===            %s\n', conds{c}, sprintf('%6s', J{:}));
  for r = 1:size(rows,1)
    x = tune_eval(rows{r,2});
    fprintf('%-15s %5.2f | %s\n', rows{r,1}, 100*x.pck, sprintf('%6.1f', 100*x.pck7));
  end
end
