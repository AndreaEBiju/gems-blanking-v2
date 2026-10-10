function check_common_start(startsFile, session, outFile)
% CHECK_COMMON_START  RULING 2026-10-10 item 7 on the Night 6 side, for
% tests/test_night6_common_start.py: the v6 starts file is read, the lead-in records each
% analysis's own AND common start beside the results (and the reference-sensitive "until"),
% and night6_common_start_rec moves every cut to its common start - and refuses a record
% without one.
    RS = night6_recovery_start(startsFile);
    F = [];
    for k = 1:numel(RS.files)
        if strcmp(RS.files{k}.session, session), F = RS.files{k}; end
    end
    i0 = 0;
    n = round(400 * F.fs);
    consumers = {'spikes', 'hrv', 'breathing', 'mmc', 'slow_wave'};
    [~, rec] = night6_recovery_lead_in(RS, session, 'stim_recovery', i0, n, F.fs, consumers, ...
                                       'mask_to_electrical_drop_outputs');
    rc = night6_common_start_rec(rec);
    R = struct('rec', jsonencode(rec), 'common', jsonencode(rc));
    bad = rec;
    bad.cuts{1} = rmfield(bad.cuts{1}, 'common_start_sample0');
    R.refused = err(@() night6_common_start_rec(bad));
    R.refused_not_applying = err(@() night6_common_start_rec(struct('applies', false)));
    fid = fopen(outFile, 'w', 'n', 'UTF-8');
    fwrite(fid, jsonencode(R), 'char');
    fclose(fid);
end

function e = err(f)
    e = struct('id', '', 'message', '');
    try
        f();
    catch ME
        e = struct('id', ME.identifier, 'message', ME.message);
    end
end
