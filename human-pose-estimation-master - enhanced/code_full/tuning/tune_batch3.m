function tune_batch3(shard, nShards)
% Abstract items still to measure: CLAHE, and adaptive HOG scale sampling
% (denser pyramid, 15 levels per octave) on the gated pipeline.
    idx = shard:nShards:205;
    all5 = {'clean','dark','lowcontrast','noise','blur'};
    tune_detect({'clahe','clahe_flip'}, all5, idx);
    tune_detect({'gated','gated_flip','gated@i15','gated_flip@i15'}, all5, idx);
end
