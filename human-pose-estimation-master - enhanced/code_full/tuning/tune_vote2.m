% Part-consistent voting radius search (tuning split 1:100, gated pipeline).
addpath(fullfile(pwd,'tuning')); addpath(pwd);
conds = {'clean','dark','lowcontrast','noise','blur'};
S = load(fullfile('cache','tuning','quality.mat')); Q = S.Q;
tags = {'base','nlm','gamma067','deconv15'}; val = 1:100;
for c = 1:5
  q = Q.(conds{c}); tag = ones(205,1);
  tag(q(:,1) < 0.40) = 3; tag(q(:,4) < 0.014) = 4; tag(q(:,3) > 6) = 2;
  pair{c} = arrayfun(@(n) {tags{tag(n)}, [tags{tag(n)} '_flip']}, val, 'UniformOutput', false);
end
for r = [0.20 0.25 0.30]
 for cfg = [0.5 0.5 0.25; 0.5 1.0 0.25; 0.5 1.0 0.5; 0.3 0.5 0.25; 0.3 1.0 0.5]'
  P = zeros(1,5); A = zeros(1,5); J = zeros(5,7);
  for c = 1:5
    s = struct('variants',{{}},'perImage',{pair{c}},'cond',conds{c},'idx',val,'refine',true, ...
               'vote',struct('iou',cfg(1),'delta',cfg(2),'T',cfg(3),'r',r));
    x = tune_eval(s); P(c) = x.pck; A(c) = x.apk; J(c,:) = x.pck7;
  end
  fprintf('r %-4g iou %.1f d %.2f T %.2f | PCK %.2f APK %.2f | %s | joints %s\n', r, cfg, 100*mean(P), 100*mean(A), ...
      mat2str(round(100*P,1)), mat2str(round(100*mean(J),1)));
 end
end
