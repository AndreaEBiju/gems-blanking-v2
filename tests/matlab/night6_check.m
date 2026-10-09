function night6_check(caseFile, outFile)
% NIGHT6_CHECK  Harness for tests/test_night6_wrapper.py (gems-blanking-v2, tests/matlab).
%
% Reads a JSON case file written by the Python test and, in ONE MATLAB process:
%   * inverts every Python-made token with night6_signal_from_token and forwards every
%     signal name with night6_token_from_signal (refusals reported as the error id);
%   * runs night6_run_recording with DryRun on each synthetic mask folder (the real
%     loading, slicing, masking and skip logic; the consumers' calls are not made);
%   * slow wave, one channel at a time: Andrea's slowWaveAnalysis_new on a small
%     three-column input, directly and through night6_keep_slow_wave;
%   * process_dataset_v2 on a small synthetic spike input, and its refusals.
% Writes what it measured as JSON; the records land under the case's out_root.
    C = jsondecode(fileread(caseFile));
    out = struct();
    out.from_token = cellfun(@(t) attempt(@night6_signal_from_token, t), C.tokens, ...
                             'UniformOutput', false);
    out.to_token = cellfun(@(s) attempt(@night6_token_from_signal, s), C.signals, ...
                           'UniformOutput', false);
    folders = cellstr(C.mask_folders);
    out.errors = cell(1, numel(folders));
    for k = 1:numel(folders)
        try
            night6_run_recording(folders{k}, 'GemsRoot', C.gems_root, 'Units', C.units, ...
                'OutRoot', C.out_root, 'DryRun', true, 'CodeCommit', 'test');
            out.errors{k} = '';
        catch ME
            out.errors{k} = sprintf('%s: %s', ME.identifier, ME.message);
        end
    end
    % Resume: mark one epoch complete (right hash), one complete for ANOTHER mask file
    % (wrong hash), leave a stale file in both, and run that mask folder again.
    if isfield(C, 'resume')
        Rz = C.resume;
        for tag = {Rz.keep_tag, Rz.stale_tag}
            d = fullfile(Rz.out_dir, tag{1});
            f = fullfile(d, 'night6_record.json');
            txt = strrep(fileread(f), '"status": "dry_run"', '"status": "complete"');
            if strcmp(tag{1}, Rz.stale_tag)
                txt = regexprep(txt, '"mask_file_sha256": "[0-9a-f]+"', ...
                                '"mask_file_sha256": "deadbeef"');
            end
            fid = fopen(f, 'w', 'n', 'UTF-8'); fwrite(fid, txt, 'char'); fclose(fid);
            fid = fopen(fullfile(d, 'stale_marker.txt'), 'w'); fwrite(fid, 'x'); fclose(fid);
        end
        try
            night6_run_recording(Rz.mask_folder, 'GemsRoot', C.gems_root, 'Units', C.units, ...
                'OutRoot', C.out_root, 'DryRun', true, 'CodeCommit', 'test');
            out.resume_error = '';
        catch ME
            out.resume_error = sprintf('%s: %s', ME.identifier, ME.message);
        end
    end
    % mmc R-peak units (RULING 2026-10-08 (g) 2): Andrea's extract_mmc, unchanged, on a
    % small synthetic input with the options the wrapper builds. It reports the R-peak
    % samples it actually used (qc.rpeakT) and the cardiac-blanked fraction.
    if isfield(C, 'mmc_units')
        U = C.mmc_units;
        try
            op = night6_mmc_opts(U.beats_file, U.fs, U.n);
            op.save = false;
            mm = extract_mmc(U.input_file, U.beats_file, op);
            out.mmc_rpeak_samples = round(mm.qc.rpeakT(:)' * U.fs);
            out.mmc_pct_blanked = mm.qc.pctBlanked;
            out.mmc_error = '';
        catch ME
            out.mmc_error = sprintf('%s: %s', ME.identifier, ME.message);
        end
    end
    if isfield(C, 'slow_wave'), out.slow_wave = slow_wave_case(C.slow_wave); end
    if isfield(C, 'v2'), out.v2 = v2_case(C.v2); end
    fid = fopen(outFile, 'w', 'n', 'UTF-8');
    fwrite(fid, jsonencode(out), 'char');
    fclose(fid);
end

function r = slow_wave_case(W)
% Her slowWaveAnalysis_new on X (three columns, channel keep's mask on all of them),
% once directly and once followed by night6_keep_slow_wave. Reports what survives.
    I = load(W.input_file);
    signals = cellstr(W.signals);
    keep = cellstr(W.keep);
    args = {logical(W.low_pass_on), W.cutoff, W.order, I.fs, W.window, false};
    r = struct('error', '');
    try
        mkdir(W.direct_dir);
        D = slowWaveAnalysis_new(I.X, args{:}, W.direct_dir, 'direct', [], W.edge_s);
        r.direct_peaks = cellfun(@(p) p(:)', D.slowWavePeakLocs, 'UniformOutput', false);
        r.direct_avg = D.avgSlowWave;
        mkdir(W.run_dir);
        label = sprintf('%s_swm_%s', W.base, W.mask_signal);
        d0 = dir(W.run_dir);
        slowWaveAnalysis_new(I.X, args{:}, W.run_dir, label, [], W.edge_s);
        d1 = dir(W.run_dir);
        r.kept = night6_keep_slow_wave(W.run_dir, label, W.base, signals, keep, ...
                                       W.mask_signal, setdiff({d1.name}, {d0.name}));
        d2 = dir(fullfile(W.run_dir, '*.mat'));
        r.files = {d2.name};
        r.kept_peaks = struct();
        r.kept_avg = struct();
        for s = keep(:)'
            K = load(fullfile(W.run_dir, sprintf('%s_slowWaves_%s.mat', W.base, s{1})));
            r.kept_peaks.(s{1}) = K.slowWavePeakLocs(:)';
            r.kept_avg.(s{1}) = K.avgSlowWave;
            r.kept_cols.(s{1}) = size(K.slowWaveTimeSeries, 2);
            r.kept_channel.(s{1}) = K.channel;
        end
    catch ME
        r.error = sprintf('%s: %s', ME.identifier, ME.message);
    end
end

function r = v2_case(V)
% process_dataset_v2 on a small synthetic D (her bulk_load_one struct) and its refusals.
    I = load(V.input_file);
    labels = cellstr(V.labels);
    D = struct('fs', I.fs, 'y', I.y, 't', (0:size(I.y, 1) - 1)' / I.fs, ...
               'removedSegmentIdx', zeros(0, 2), 'neuralChannels', 1:size(I.y, 2), ...
               'channelLabels', {labels}, 'rpeakSamples', I.rpeakSamples(:), ...
               'rpeakTimes', (I.rpeakSamples(:) - 1) / I.fs);
    r = struct('error', '');
    try
        [Dv, info] = process_dataset_v2(D);
        r.steps = cellfun(@(s) s.name, info.steps, 'UniformOutput', false);
        r.step_dirs = cellfun(@(s) fileparts(s.path), info.steps, 'UniformOutput', false);
        r.spike_check = info.spike_check;
        r.n_rpeaks = info.n_rpeaks;
        r.bandpass = [info.P.bandpassLow, info.P.bandpassHigh, info.P.threshSigma];
        r.centers = arrayfun(@(s) s.alignedCenters(:)', Dv.spikes, 'UniformOutput', false);
        r.input_nan_still_nan = all(isnan(Dv.y(isnan(D.y))));
        r.step1a_ran = isfield(Dv, 'cardiacBlank');   % ruling (i): it must not
        r.rpeak_guard_ms = Dv.envelope(1).guardMs;    % step3b kept its guard
    catch ME
        r.error = sprintf('%s: %s', ME.identifier, ME.message);
    end
    % A pad wider than her own 10 ms + 0.5 ms re-alignment must fire: the check is live.
    r.wide_pad = attempt_id(@() process_dataset_v2(D, 'NanPadMs', V.wide_pad_ms));
    E = D; E.y(:, 2) = NaN;
    r.all_nan = attempt_id(@() process_dataset_v2(E));
    E = D; E.y(isfinite(E.y(:, 2)), 2) = 3e-6;
    r.constant = attempt_id(@() process_dataset_v2(E));
    E = D; E.rpeakSamples(1) = E.rpeakSamples(1) + 0.5;
    r.bad_rpeaks = attempt_id(@() process_dataset_v2(E));
end

function id = attempt_id(f)
    try
        f();
        id = '';
    catch ME
        id = ME.identifier;
    end
end

function r = attempt(f, x)
    try
        r = f(x);
    catch ME
        r = ['!' ME.identifier];
    end
end
