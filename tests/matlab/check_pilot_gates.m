function check_pilot_gates(caseFile, outFile)
% CHECK_PILOT_GATES  Harness for tests/test_night6_wrapper.py: review 2026-10-10 fixes 1-3.
%
% For each run case, night6_run_recording (DryRun) on a mask folder of the synthetic store
% with the case's OutRoot, PilotRoot and BeatsRoot; a case with move_store true runs with the
% store's own beats file moved away (restored after). For each batch case, night6_batch
% (DryRun, 0 workers) on the case's list file; a batch that returns reports its recordings'
% statuses. Reports '' or '<identifier>: <message>' per case.
    C = jsondecode(fileread(caseFile));
    common = {'GemsRoot', C.gems_root, 'Units', C.units, 'DryRun', true, ...
              'CodeCommit', 'test', 'RecoveryTrimMode', 'mask_to_electrical_drop_outputs', ...
              'SlowWaveRate', 'full'};
    out = struct('run', struct(), 'batch', struct());
    cases = C.run_cases;
    if ~iscell(cases), cases = num2cell(cases); end
    for k = 1:numel(cases)
        K = cases{k};
        moved = false;
        if K.move_store
            movefile(C.store_beats, [C.store_beats '.moved']);
            moved = true;
        end
        try
            night6_run_recording(K.mask_folder, common{:}, 'OutRoot', K.out_root, ...
                                 'PilotRoot', K.pilot_root, 'BeatsRoot', K.beats_root);
            r = '';
        catch ME
            r = sprintf('%s: %s', ME.identifier, ME.message);
        end
        if moved, movefile([C.store_beats '.moved'], C.store_beats); end
        out.run.(K.name) = r;
    end
    lists = C.batch_cases;
    if ~iscell(lists), lists = num2cell(lists); end
    for k = 1:numel(lists)
        K = lists{k};
        try
            s = night6_batch(K.list_file, 0, 'DryRun', true);
            st = cellfun(@(x) x.status, s.recordings, 'UniformOutput', false);
            r = ['returned: ' strjoin(st, ' | ')];
        catch ME
            r = sprintf('%s: %s', ME.identifier, ME.message);
        end
        out.batch.(K.name) = r;
    end
    fid = fopen(outFile, 'w', 'n', 'UTF-8');
    fwrite(fid, jsonencode(out), 'char');
    fclose(fid);
end
