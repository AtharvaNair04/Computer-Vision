% Grid-search pose-voting parameters on validation (idx 1:100), base+flip pool.
addpath(fullfile(pwd,'tuning')); addpath(pwd);
conds = {'clean','dark','lowcontrast','noise','blur'};
best = -inf;
for iou = [0.3 0.5 0.7]
 for delta = [0.25 0.5 1.0]
  for T = [0.1 0.25 0.5]
   P = zeros(1,5); A = zeros(1,5);
   for c = 1:5
     s = struct('variants',{{'base','base_flip'}},'cond',conds{c},'idx',1:100, ...
                'vote',struct('iou',iou,'delta',delta,'T',T));
     r = tune_eval(s); P(c) = r.pck; A(c) = r.apk;
   end
   fprintf('iou %.1f delta %.2f T %.2f | PCK %.2f APK %.2f | %s\n', iou, delta, T, 100*mean(P), 100*mean(A), mat2str(round(100*P,1)));
  end
 end
end
