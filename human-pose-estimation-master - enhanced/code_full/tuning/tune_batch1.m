function tune_batch1(shard, nShards)
% Phase-1 detection sweep, split over nShards MATLAB processes.
    idx = shard:nShards:205;
    all5 = {'clean','dark','lowcontrast','noise','blur'};
    tune_detect({'base','base_flip','adapt_v1'}, all5, idx);
    tune_detect({'gauss07','gauss10','median3','wiener5','bilat_strong','nlm'}, {'noise','clean'}, idx);
    tune_detect({'gamma067','stretch'}, {'dark','lowcontrast','clean'}, idx);
    tune_detect({'deconv15','sharp_mild'}, {'blur','clean'}, idx);
    tune_detect({'up125','up125_flip'}, all5, idx);
end
