function check_no_valid_window(caseFile, outFile)
% CHECK_NO_VALID_WINDOW  Harness for tests/test_night6_slow_wave_no_valid_window.py.
%
% A REAL (not dry) Night 6 slow-wave run on a synthetic store epoch too short for any rate
% window (RULING 2026-10-13: "no valid window", never held or failed); writes the record path,
% or the error.
    C = jsondecode(fileread(caseFile));
    addpath(C.processing_new); addpath(C.night6);
    out = struct('error', '', 'record', '');
    try
        R = night6_run_recording(C.mask_folder, 'GemsRoot', C.gems_root, 'Units', 'uV', ...
            'OutRoot', C.out_root, 'CodeCommit', 'test', ...
            'RecoveryTrimMode', 'mask_to_electrical_drop_outputs', ...
            'SlowWaveRate', 'decimated78', 'Consumers', {'slow_wave'});
        out.status = R{1}.status;
        f = dir(fullfile(C.out_root, '**', 'night6_record.json'));
        out.record = fullfile(f(1).folder, f(1).name);
    catch ME
        out.error = sprintf('%s: %s', ME.identifier, ME.message);
    end
    fid = fopen(outFile, 'w', 'n', 'UTF-8');
    fwrite(fid, jsonencode(out), 'char');
    fclose(fid);
end
