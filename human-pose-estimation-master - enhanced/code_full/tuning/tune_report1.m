% Phase-1 validation report (test indices 1:100 only).
addpath(fullfile(pwd,'tuning')); addpath(pwd);
val = 1:100;
rows = {
 'clean',       {'base'},{'base_flip'},{'base','base_flip'},{'adapt_v1'},{'up125'},{'base','up125'},{'gauss07'},{'median3'},{'gamma067'},{'stretch'},{'deconv15'},{'sharp_mild'}
 'dark',        {'base'},{'base_flip'},{'base','base_flip'},{'adapt_v1'},{'up125'},{'base','up125'},{'gamma067'},{'stretch'},{'base','gamma067'},{},{},{}
 'lowcontrast', {'base'},{'base_flip'},{'base','base_flip'},{'adapt_v1'},{'up125'},{'base','up125'},{'gamma067'},{'stretch'},{'base','stretch'},{},{},{}
 'noise',       {'base'},{'base_flip'},{'base','base_flip'},{'adapt_v1'},{'up125'},{'gauss07'},{'gauss10'},{'median3'},{'wiener5'},{'bilat_strong'},{'nlm'},{'base','nlm'}
 'blur',        {'base'},{'base_flip'},{'base','base_flip'},{'adapt_v1'},{'up125'},{'base','up125'},{'deconv15'},{'sharp_mild'},{'base','deconv15'},{},{},{}
};
for r = 1:size(rows,1)
  fprintf('=== %s ===\n', rows{r,1});
  for k = 2:size(rows,2)
    v = rows{r,k}; if isempty(v), continue; end
    s = struct('variants',{v},'cond',rows{r,1},'idx',val);
    a = tune_eval(s);
    s.vote = struct('iou',0.5,'delta',0.5,'T',0.25); b = tune_eval(s);
    fprintf('%-22s PCK %.2f APK %.2f | +vote PCK %.2f APK %.2f\n', strjoin(v,'+'), 100*a.pck, 100*a.apk, 100*b.pck, 100*b.apk);
  end
end
