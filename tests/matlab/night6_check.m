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
%   * slow wave with two channels kept from one shared call;
%   * night6_consumer_input's joint-mask assertion on a hand-built plan;
%   * the step1a fallback list's hash (raw bytes) on LF, CRLF and non-ASCII files;
%   * process_dataset_v2 on a small synthetic spike input, its refusals, and two
%     mutants from tests/matlab/mutants (a zero-filling step 1; threshSigma 5), each
%     on the path only for its own call.
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
                'OutRoot', C.out_root, 'DryRun', true, 'CodeCommit', 'test', ...
                'RecoveryTrimMode', 'mask_to_electrical_drop_outputs');
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
                'OutRoot', C.out_root, 'DryRun', true, 'CodeCommit', 'test', ...
                'RecoveryTrimMode', 'mask_to_electrical_drop_outputs');
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
    % Ruling (j) 1: the declared step1a fallback list - refusals, and a run that uses it.
    if isfield(C, 'fallback')
        Fb = C.fallback;
        out.fallback_bad = cellfun(@(f) attempt_id(@() night6_step1a_fallback(f)), ...
                                   cellstr(Fb.bad_files), 'UniformOutput', false);
        try
            night6_run_recording(Fb.mask_folder, 'GemsRoot', C.gems_root, 'Units', C.units, ...
                'OutRoot', Fb.out_root, 'DryRun', true, 'CodeCommit', 'test', ...
                'RecoveryTrimMode', 'mask_to_electrical_drop_outputs', ...
                'Step1aFallback', Fb.file_ok);
            out.fallback_error = '';
        catch ME
            out.fallback_error = sprintf('%s: %s', ME.identifier, ME.message);
        end
    end
    % The fallback list's hash is of the raw bytes (night6_sha256_file), so an LF file,
    % its CRLF copy and a non-ASCII file each hash as Python's hashlib does.
    if isfield(C, 'fallback_hash')
        out.fallback_sha = cellfun(@fallback_sha, cellstr(C.fallback_hash), ...
                                   'UniformOutput', false);
    end
    out.joint = joint_mask_case();
    if isfield(C, 'slow_wave'), out.slow_wave = slow_wave_case(C.slow_wave); end
    if isfield(C, 'slow_wave2'), out.slow_wave2 = slow_wave_case(C.slow_wave2); end
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
        % Invariant 1 on what the steps WROTE (columns = neural channels 1:2 here).
        r.input_nan_filtered_nan = all(isnan(Dv.filtered(isnan(D.y))));
        r.input_nan_invalid = ~any(Dv.validMask(isnan(D.y)));
        r.n_input_nan = nnz(isnan(D.y));
        r.info_P = struct('threshSigma', info.P.threshSigma, ...
                          'refractoryMs', info.P.refractoryMs, ...
                          'detectPolarity', info.P.detectPolarity);
        r.step1a_ran = isfield(Dv, 'cardiacBlank');   % ruling (i): it must not
        r.rpeak_guard_ms = Dv.envelope(1).guardMs;    % step3b's guard: 0, RULING (k) 3
    catch ME
        r.error = sprintf('%s: %s', ME.identifier, ME.message);
    end
    % A pad wider than her own 10 ms + 0.5 ms re-alignment must fire: the check is live.
    r.wide_pad = attempt_id(@() process_dataset_v2(D, 'NanPadMs', V.wide_pad_ms));
    % Ruling (j) 1 fallback: her step1a on channel 1 only.
    try
        [Df, fi] = process_dataset_v2(D, 'Step1aChannels', [true false]);
        r.fb_channels = fi.step1a_channels;
        r.fb_nan_added = [nnz(isnan(Df.y(:, 1))) - nnz(isnan(D.y(:, 1))), ...
                          nnz(isnan(Df.y(:, 2))) - nnz(isnan(D.y(:, 2)))];
        r.fb_spike_check = fi.spike_check;
        r.fb_error = '';
    catch ME
        r.fb_error = sprintf('%s: %s', ME.identifier, ME.message);
    end
    E = D; E.rpeakSamples = zeros(0, 1); E.rpeakTimes = zeros(0, 1);
    r.fb_no_rpeaks = attempt_id(@() process_dataset_v2(E, 'Step1aChannels', [true false]));
    E = D; E.y(:, 2) = NaN;
    r.all_nan = attempt_id(@() process_dataset_v2(E));
    E = D; E.y(isfinite(E.y(:, 2)), 2) = 3e-6;
    r.constant = attempt_id(@() process_dataset_v2(E));
    E = D; E.rpeakSamples(1) = E.rpeakSamples(1) + 0.5;
    r.bad_rpeaks = attempt_id(@() process_dataset_v2(E));
    % Mutants: a step 1 that zero-fills NaN, and her default threshold changed. Each is
    % on the path only for its own call; which() proves the mutant was the one called.
    [r.mutant_zero_fill, r.mutant_zero_fill_path] = with_mutant('step1_zero_fill', ...
        'step1_bandpass', @() process_dataset_v2(D));
    [r.mutant_thresh, r.mutant_thresh_path] = with_mutant('thresh_sigma_5', ...
        'pipeline_params', @() process_dataset_v2(D));
    r.after_mutants_step1 = which('step1_bandpass');   % back to hers
end

function [id, where] = with_mutant(dirName, fname, f)
% Run f with tests/matlab/mutants/<dirName> first on the path; report the error id and
% where fname resolved while it was there. The path is restored whatever happens.
    d = fullfile(fileparts(mfilename('fullpath')), 'mutants', dirName);
    addpath(d, '-begin');
    restore = onCleanup(@() rmpath(d));
    where = which(fname);
    id = attempt_id(f);
    clear restore
end

function h = fallback_sha(f)
    F = night6_step1a_fallback(f);
    h = F.sha256;
end

function r = joint_mask_case()
% night6_consumer_input on a hand-built 12-row plan: a slow-wave run on ANT1's mask is
% accepted when the joint NaN mask IS ANT1's mask, and refused by name
% ('night6:jointMask') when the recording itself carries a NaN elsewhere.
    plan = struct('i0', 2, 'n', 12, 'scaleToVolts', 1);
    plan.recipes = struct();
    for c = 1:3
        plan.recipes.(sprintf('ANT%d', c)) = struct('kind', 'raw', 'cols', c, 'weights', 1);
    end
    plan.masks = struct('consumer', {'slow_wave', 'slow_wave', 'slow_wave'}, ...
                        'signal', {'ANT1', 'ANT2', 'ANT3'}, 'token', {'ANT1', 'ANT2', 'ANT3'}, ...
                        'spans', {[3 4; 9 9], zeros(0, 2), [1 1]});
    run = struct('call', 'slowWaveAnalysis_new', 'consumers', {{'slow_wave'}}, ...
                 'signals', {{'ANT1', 'ANT2', 'ANT3'}}, 'maskSignal', 'ANT1', 'keep', {{'ANT1'}});
    Y = reshape(1:16 * 3, 16, 3);
    r = struct();
    try
        X = night6_consumer_input(plan, Y, run);
        r.ok_error = '';
        r.ok_nan_rows = find(any(isnan(X), 2))';
    catch ME
        r.ok_error = ME.identifier;
    end
    Y(2 + 6, 2) = NaN;   % epoch row 6 of ANT2: not in ANT1's mask
    r.stray_nan = attempt_id(@() night6_consumer_input(plan, Y, run));
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
