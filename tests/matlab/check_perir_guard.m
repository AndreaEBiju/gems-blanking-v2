function check_perir_guard(caseFile, outFile)
% CHECK_PERIR_GUARD  Harness for tests/test_peri_r.py (gems-blanking-v2, tests/matlab).
%
% RULING 2026-10-08 (k) 3: process_dataset_v2 runs Andrea's steps (read only, from
% processing_new) with night6_v2_params, whose step3b guard is 0. For each case - a
% synthetic two-channel input (volts) whose peri-R spans are already NaN, and its 1-based
% R-peaks - it records, per channel, step3b's valid fraction per RMS bin
% (D.envelope(k).validFrac), the bin length in samples, and the guard and edge pad that
% were actually used. The Python test turns the valid fractions back into excluded-sample
% counts and compares them with span + pad, bin by bin.
    C = jsondecode(fileread(caseFile));
    labels = cellstr(C.labels);
    out = struct();
    for name = fieldnames(C.cases)'
        I = load(C.cases.(name{1}));
        D = struct('fs', I.fs, 'y', I.y, 't', (0:size(I.y, 1) - 1)' / I.fs, ...
                   'removedSegmentIdx', zeros(0, 2), 'neuralChannels', 1:size(I.y, 2), ...
                   'channelLabels', {labels}, 'rpeakSamples', I.rpeakSamples(:), ...
                   'rpeakTimes', (I.rpeakSamples(:) - 1) / I.fs);
        r = struct('error', '');
        try
            [Dv, info] = process_dataset_v2(D);
            r.guard_ms = info.P.envCardiacGuardMs;
            r.edge_buffer_ms = info.P.edgeBufferMs;
            r.bin_n = max(1, round(info.P.envBinSec * I.fs));
            r.valid_frac = arrayfun(@(e) e.validFrac(:)', Dv.envelope, 'UniformOutput', false);
            r.envelope_guard_ms = Dv.envelope(1).guardMs;
        catch ME
            r.error = sprintf('%s: %s', ME.identifier, ME.message);
        end
        out.(name{1}) = r;
    end
    fid = fopen(outFile, 'w');
    fwrite(fid, jsonencode(out), 'char');
    fclose(fid);
end
