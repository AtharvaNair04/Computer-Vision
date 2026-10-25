function tune_batch2(shard, nShards)
% Phase-2: flipped versions of the best per-condition transforms.
    idx = shard:nShards:205;
    tune_detect({'nlm_flip'}, {'noise','clean'}, idx);
    tune_detect({'gamma067_flip'}, {'dark','clean','lowcontrast'}, idx);
    tune_detect({'deconv15_flip'}, {'blur','clean'}, idx);
    tune_detect({'nlm','nlm_flip','gamma067','gamma067_flip','deconv15','deconv15_flip'}, {'dark','lowcontrast','noise','blur'}, idx);
end
