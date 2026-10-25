% Does anatomical refinement still help on top of flip + voting? (validation)
addpath(fullfile(pwd,'tuning')); addpath(pwd);
conds = {'clean','dark','lowcontrast','noise','blur'};
vote = struct('iou',0.7,'delta',0.25,'T',0.1);
cfgs = {struct(), struct('symmetryWeight',0), struct('symmetryWeight',0.2), ...
        struct('maxUpperArmRatio',1.0,'maxForearmRatio',0.9,'maxThighRatio',1.3,'maxCalfRatio',1.25)};
names = {'none','default','noSym','sym0.2','looser'};
for k = 0:numel(cfgs)
  P = zeros(1,5); A = zeros(1,5);
  for c = 1:5
    s = struct('variants',{{'base','base_flip'}},'cond',conds{c},'idx',1:100,'vote',vote);
    if k > 0, s.refine = true; s.refineParams = cfgs{k}; end
    r = tune_eval(s); P(c) = r.pck; A(c) = r.apk;
  end
  fprintf('%-8s PCK %.2f APK %.2f | %s\n', names{k+1}, 100*mean(P), 100*mean(A), mat2str(round(100*P,1)));
end
