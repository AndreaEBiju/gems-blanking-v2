function check_recovery_start(caseFile, outFile)
% CHECK_RECOVERY_START  Harness for tests/test_night6_recovery_start.py (RULING 2026-10-08 (k) 2).
%
%   plans    each case plans ONE epoch from a mask file written by the real
%            emit.handoff.write_mask_file, twice: without 'Recovery' (the untrimmed
%            baseline) and with the case's starts file. For both it reports, per run and
%            per input column, the NaN rows of night6_consumer_input (1-based inclusive
%            runs), plus plan.recoveryStart - or the error identifier and message.
%   readers  night6_recovery_start on each file: 'ok' or the error identifier.
%   run      night6_run_recording (DryRun) on a store mask folder: without RecoveryStarts,
%            with one, without / with an unknown RecoveryTrimMode, and the resume rule (a
%            complete record made with another starts file, or another trim mode, reruns;
%            one made with this file and mode is skipped).
%   trim     the drop mode end to end with her real functions, against an untrimmed
%            reference run on the same masked input, through an oracle with its own
%            convention table (trim_case, oracle_table), and the marker in every file.
%   unit     night6_recovery_trim_outputs on hand-built files, every convention at L-1/L/L+1.
%   sources  the reader against a changed copy of a cited function on the path.
%   batch    night6_batch's refusals at batch start.
    C = jsondecode(fileread(caseFile));
    out = struct();
    out.plans = cellfun(@plan_case, as_cells(C.plans), 'UniformOutput', false);
    out.readers = cellfun(@reader_case, as_cells(C.readers), 'UniformOutput', false);
    if isfield(C, 'run'), out.run = run_case(C.run); end
    if isfield(C, 'unit'), out.unit = unit_case(C.unit); end
    if isfield(C, 'batch'), out.batch = batch_case(C.batch); end
    if isfield(C, 'trim'), out.trim = trim_case(C.trim); end
    if isfield(C, 'sources'), out.sources = sources_case(C.sources); end
    fid = fopen(outFile, 'w', 'n', 'UTF-8');
    fwrite(fid, jsonencode(out), 'char');
    fclose(fid);
end

function r = plan_case(K)
    r = struct('name', K.name, 'error', '', 'message', '', 'base', {{}}, 'trim', {{}}, ...
               'record', '');
    try
        M = load(K.mask_file);
        meta = jsondecode(K.meta_json);
        beats = [];
        if isfield(K, 'beats_file') && ~isempty(K.beats_file)
            beats = struct('data', load(K.beats_file), 'record', jsondecode(K.record_json), ...
                           'sha256', night6_sha256_file(K.beats_file));
        end
        Y = ones(K.n_file, numel(K.labels));
        base = night6_prepare_epoch(M, K.labels, meta.channels, K.n_file, K.fs, 'uV', beats, ...
                                    K.condition);
        r.base = describe(base, Y);
        RS = night6_recovery_start(K.starts_file);
        plan = night6_prepare_epoch(M, K.labels, meta.channels, K.n_file, K.fs, 'uV', beats, ...
                                    K.condition, 'Recovery', struct('starts', RS, ...
                                                                    'session', K.session, ...
                                                                    'mode', K.mode));
        r.trim = describe(plan, Y);
        r.record = jsonencode(plan.recoveryStart);
    catch ME
        r.error = ME.identifier;
        r.message = ME.message;
    end
end

function d = describe(plan, Y)
    d = {};
    for run = plan.runs
        X = night6_consumer_input(plan, Y, run);
        for j = 1:size(X, 2)
            b = isnan(X(:, j));
            e = diff([false; b; false]);
            d{end + 1} = struct('call', run.call, 'consumers', {run.consumers}, ...
                                'signal', run.signals{j}, 'mask_signal', run.maskSignal, ...
                                'keep', {run.keep}, ...
                                'nan_runs', [find(e == 1), find(e == -1) - 1]); %#ok<AGROW>
        end
    end
end

function r = reader_case(K)
    r = struct('name', K.name, 'error', '', 'message', '');
    try
        night6_recovery_start(K.file);
    catch ME
        r.error = ME.identifier;
        r.message = ME.message;
    end
end

function r = run_case(K)
    args0 = {'GemsRoot', K.gems_root, 'Units', 'uV', 'DryRun', true, 'CodeCommit', 'test'};
    args = [args0, {'RecoveryTrimMode', 'mask_to_own_start'}];
    r = struct();
    r.without = attempt(@() night6_run_recording(K.mask_folder, 'OutRoot', K.out_a, args{:}));
    r.no_mode = attempt(@() night6_run_recording(K.mask_folder, 'OutRoot', K.out_a, ...
        'RecoveryStarts', K.starts_file, args0{:}));
    r.bad_mode = attempt(@() night6_run_recording(K.mask_folder, 'OutRoot', K.out_a, ...
        'RecoveryStarts', K.starts_file, 'RecoveryTrimMode', 'mask_everything', args0{:}));
    R = night6_run_recording(K.mask_folder, 'OutRoot', K.out_b, 'RecoveryStarts', ...
                             K.starts_file, args{:});
    r.with = cellfun(@(x) jsonencode(x), R, 'UniformOutput', false);
    % resume: a COMPLETE record made with this starts file is skipped; with another, rerun
    recFile = fullfile(K.out_b, K.record_rel);
    Rc = jsondecode(fileread(recFile));
    Rc.status = 'complete';
    write(recFile, Rc);
    R2 = night6_run_recording(K.mask_folder, 'OutRoot', K.out_b, 'RecoveryStarts', ...
                              K.starts_file, args{:});
    r.resume_same = R2{1}.status;
    R3 = night6_run_recording(K.mask_folder, 'OutRoot', K.out_b, 'RecoveryStarts', ...
                              K.other_starts_file, args{:});
    r.resume_other = R3{1}.status;
    % the same starts file under ANOTHER trim mode: rerun
    Rc = jsondecode(fileread(recFile));
    Rc.status = 'complete';
    write(recFile, Rc);
    R4 = night6_run_recording(K.mask_folder, 'OutRoot', K.out_b, 'RecoveryStarts', ...
                              K.other_starts_file, args0{:}, 'RecoveryTrimMode', ...
                              'mask_to_electrical_drop_outputs');
    r.resume_other_mode = R4{1}.status;
    r.mode_recorded = R4{1}.recovery_trim_mode;
end

function s = attempt(f)
    s = struct('error', '', 'message', '');
    try
        f();
    catch ME
        s.error = ME.identifier;
        s.message = ME.message;
    end
end

function write(f, R)
    fid = fopen(f, 'w', 'n', 'UTF-8');
    fwrite(fid, jsonencode(R), 'char');
    fclose(fid);
end

function c = as_cells(v)
    if isempty(v), c = {}; elseif isstruct(v), c = num2cell(v(:))'; else, c = v(:)'; end
end

% ==========================================================================
% (k) 2 trim modes (review of be402a1)

function r = trim_case(K)
% The drop mode end to end, with her real functions: the same epoch run twice on the
% same masked input (every input masked to the electrical settling) - once with the
% drop mode (outputs cut at each own start), once as the untrimmed reference
% (mask_to_own_start with every own start AT the electrical settling: no output cut).
% Every output file is compared through an ORACLE that does not read the map: its own
% table (oracle_table, transcribed from her code with file:line) says which variables
% are time-stamped, by what, in which convention, with which action and owner. Each
% entry is cut at its OWNER's start (K.L, every analysis), whether or not the owner ran.
% The marker each trimmed file carries is checked against the same oracle, exactly.
    args = {'GemsRoot', K.gems_root, 'Units', 'uV', 'CodeCommit', 'test'};
    rng(0, 'twister');
    Rd = night6_run_recording(K.mask_folder, 'OutRoot', K.out_drop, 'RecoveryStarts', ...
                              K.starts_drop, 'RecoveryTrimMode', ...
                              'mask_to_electrical_drop_outputs', args{:});
    rng(0, 'twister');
    Rr = night6_run_recording(K.mask_folder, 'OutRoot', K.out_ref, 'RecoveryStarts', ...
                              K.starts_ref, 'RecoveryTrimMode', 'mask_to_own_start', args{:});
    r = struct('record_drop', jsonencode(Rd{1}), 'record_ref', jsonencode(Rr{1}), ...
               'files', {{}}, 'vars', {{}});
    RS = night6_recovery_start(K.starts_drop);
    OT = RS.outputTimes;
    TB = oracle_table();
    r.table_vs_map = table_vs_map(TB, OT);
    r.marker_variable = OT.markerVariable;
    dD = fullfile(K.out_drop, K.epoch_rel);
    dR = fullfile(K.out_ref, K.epoch_rel);
    files = dir(fullfile(dD, '*.mat'));
    for k = 1:numel(files)
        name = files(k).name;
        T = load(fullfile(dD, name));
        U = load(fullfile(dR, name));
        hit = cellfun(@(f) ~isempty(regexp(name, f.pattern, 'once')), OT.files);
        fr = struct('file', name, 'kind', '', 'rest_equal', false, 'has_marker', false, ...
                    'marker_extra', {{}});
        if ~any(hit)
            fr.rest_equal = isequaln(T, U);
            r.files{end + 1} = fr;
            continue
        end
        kind = OT.files{hit}.kind;
        fr.kind = kind;
        fr.has_marker = isfield(T, OT.markerVariable);
        M = struct('vars', {{}});
        if fr.has_marker, M = T.(OT.markerVariable); end
        mine = TB(strcmp({TB.kind}, kind));
        Tr = T; Ur = U;
        seen = {};
        for j = 1:numel(mine)
            [res, Tr, Ur] = oracle(T, U, Tr, Ur, mine(j), K, M);
            res.file = name;
            r.vars{end + 1} = res;
            if ~strcmp(res.why, 'absent'), seen{end + 1} = mine(j).path; end %#ok<AGROW>
        end
        mpaths = cellfun(@(v) v.path, M.vars, 'UniformOutput', false);
        fr.marker_extra = setdiff(mpaths, seen);   % the marker names nothing it did not cut
        if strcmp(kind, 'mmc')   % her qc.srcFile names the input file: the out folder
            Tr.mmc.qc.srcFile = ''; Ur.mmc.qc.srcFile = '';
        end
        if fr.has_marker, Tr = rmfield(Tr, OT.markerVariable); end
        fr.rest_equal = isequaln(Tr, Ur);   % everything but the trimmed leaves: untouched
        r.files{end + 1} = fr;
    end
end

function TB = oracle_table()
% THE ORACLE'S OWN TABLE of every time-stamped output Night 6 cuts, read from Andrea's
% code (processing_new at the SHA-256s of extent.recovery_start.SOURCE_FILES), NOT from
% the map the trimmer applies. Conventions (0-based epoch position p of a stamp v):
%   row1  a 1-based sample index of the epoch:               p = v - 1
%   sec0  seconds with row 1 at t = 0, t = (row - 1) / fs:    p = v * fs
%   sec1  seconds = 1-based row / fs (one sample later):      p = v * fs - 1
%   delay extract_mmc delay_t, (lo + hi) / 2 * S in RATE ROWS: the window's true centre
%         is (rate_t(lo) + rate_t(hi)) / 2 = W/2 + ((lo + hi) / 2 - 1) * S
%                                                             p = (v - S + W/2) * fs
% Owner = the analysis whose start cuts the entry (an HR call writes both hrv's and
% breathing's outputs). Action: nan (value NaN, stamp kept), drop (removed with its
% co-indexed lists), false (logical event series).
    TB = struct('kind', {}, 'path', {}, 'stamp', {}, 'conv', {}, 'action', {}, 'owner', {});
    % --- spikes: process_dataset_v2 (night6_run_recording save_spikes_v2: her D fields)
    % step2_noise_sigma.m:89-98 window i0 = (w-1)*step+1 (1-based), cWin(w) = (i0+i1)/2;
    % :144 D.sigmaWin.centers / .sigma, a cell per channel
    TB = add(TB, 'spikes_v2', 'sigmaWin.sigma', 'sigmaWin.centers', 'row1', 'nan', 'spikes');
    % step3_detect.m:86-90 centers = locs (findpeaks rows of the epoch), times = (locs-1)/fs
    for v = {'centers', 'times', 'peakAmp_uv', 'threshAtSpike_uv', 'artifactMask'}
        TB = add(TB, 'spikes_v2', ['spikes.' v{1}], 'spikes.centers', 'row1', 'drop', 'spikes');
    end
    % step4_waveforms.m:114-118 alignedCenters (rows), alignedTimes = (aligned-1)/fs
    for v = {'waveforms', 'alignedCenters', 'alignedTimes', 'Vpp_uv', 'width_ms'}
        TB = add(TB, 'spikes_v2', ['spikes.' v{1}], 'spikes.alignedCenters', 'row1', 'drop', ...
                 'spikes');
    end
    % step3b_envelope.m:95-107 bins i0 = (b-1)*binN+1, t_c(b) = ((i0+i1)/2 - 1)/fs; :114-118
    for v = {'rms_uv', 'sigmaFloor_uv', 'excess_uv', 'validFrac'}
        TB = add(TB, 'spikes_v2', ['envelope.' v{1}], 'envelope.t', 'sec0', 'nan', 'spikes');
    end
    % step6_spike_report.m:62, firing_rate :142-151 t(b) = ((i0+i1)/2-1)/fs
    for v = {'fr_hz', 'fr_validFrac'}
        TB = add(TB, 'spikes_v2', ['metrics.' v{1}], 'metrics.fr_t', 'sec0', 'nan', 'spikes');
    end
    % step6_spike_report.m:94, rolling_cv2 :262-264 edges = 0:winSec:Tend on st, the sorted
    % alignedTimes (:34, step4:116, seconds from row 1): t = edges(1:end-1) + winSec/2
    TB = add(TB, 'spikes_v2', 'metrics.cv2_roll', 'metrics.cv2_t', 'sec0', 'nan', 'spikes');
    % step6_spike_report.m:249-252 onsets/offsets = st(i0)/st(i1); a burst is cut by its onset
    for v = {'onsets', 'offsets'}
        TB = add(TB, 'spikes_v2', ['metrics.burst.' v{1}], 'metrics.burst.onsets', 'sec0', ...
                 'drop', 'spikes');
    end
    % --- HR_BR_HRVAnalysis_beats: _HRBR.mat (save :663-674), _HRVMeasures.mat (:685-694)
    % :213 t = (0:N-1)'/fs; :303 heartBeatSeries(invalidMask) = NaN, one row per sample
    TB = add(TB, 'HRBR', 'heartBeatSeries', 't', 'sec0', 'nan', 'hrv');
    % :292-299 heartlocs: 1-based sample indices 1..N, filtered
    for f = {'HRBR', 'HRVMeasures'}
        TB = add(TB, f{1}, 'heartlocs', 'heartlocs', 'row1', 'drop', 'hrv');
    end
    % :792 metrics_t = (0:stepSec:sigDurSec)', :823-824 idxCtr = round(tc*fs)+1;
    % :849, :893-896 heart rate and counts (hrv); :866 breath rate (breathing)
    for v = {'heartRateSeries', 'heartCountSeries', 'heartCountValidSec', 'heartCountRateSeries'}
        TB = add(TB, 'HRBR', v{1}, 'metrics_t', 'sec0', 'nan', 'hrv');
    end
    TB = add(TB, 'HRBR', 'breathRateSeries', 'metrics_t', 'sec0', 'nan', 'breathing');
    % :388 br_locs_true = heartlocs(br_locs), :391 filtered: sample indices (breathing)
    TB = add(TB, 'HRBR', 'br_locs_true', 'br_locs_true', 'row1', 'drop', 'breathing');
    % :310, computeValidRRIntervals :983-1000: RR_times = s1/fs, s1 = heartlocs(i) 1-based
    for v = {'RR_intervals', 'RR_times'}
        TB = add(TB, 'HRVMeasures', v{1}, 'RR_times', 'sec1', 'drop', 'hrv');
    end
    % :901-929 windowed HRV series on metrics_t (:792)
    for v = {'hrv_series', 'rmssd_series', 'pnn5_series', 'sd1_series', 'sd2_series', ...
             'sampEn_series', 'nRR_used'}
        TB = add(TB, 'HRVMeasures', v{1}, 'metrics_t', 'sec0', 'nan', 'hrv');
    end
    % --- slowWaveAnalysis_new (save :340-345; one file per kept channel)
    % :89 t = (0:N-1)'/fs; :160 slowWaveTimeSeries(invalidMask,:) = NaN
    TB = add(TB, 'slowWaves', 'slowWaveTimeSeries', 't', 'sec0', 'nan', 'slow_wave');
    % :176-178 rateT_idx = 1:rateStepSamp:N, slowWaveRateTime = t(rateT_idx); :279
    TB = add(TB, 'slowWaves', 'slowWaveRateSeries', 'slowWaveRateTime', 'sec0', 'nan', ...
             'slow_wave');
    % :203-205 slowWavePeakLocs{ci} = locs (sample indices, a cell per channel)
    TB = add(TB, 'slowWaves', 'slowWavePeakLocs', 'slowWavePeakLocs', 'row1', 'drop', ...
             'slow_wave');
    % --- extract_mmc (save :170; struct :151-166)
    % :65 t = (0:N-1).'/fs; :153 signal = single(cond), one row per sample
    TB = add(TB, 'mmc', 'mmc.signal', 'mmc.t', 'sec0', 'nan', 'mmc');
    % :155-157 events = ev_bool (:299-303), N x 3 logical, row = sample
    for lvl = {'firing', 'burst'}
        TB = add(TB, 'mmc', ['mmc.' lvl{1} '.events'], 'mmc.t', 'sec0', 'false', 'mmc');
        % :115 centers = (W/2 : S : t(end)-W/2) in seconds of t; :154; :292-293
        for v = {'rate', 'peakAmp'}
            TB = add(TB, 'mmc', ['mmc.' lvl{1} '.' v{1}], 'mmc.rate_t', 'sec0', 'nan', 'mmc');
        end
    end
    % :118 xchan_delay on the firing rate; :259-265 delay_t = (lo+hi)/2*S in rate rows; :272
    TB = add(TB, 'mmc', 'mmc.delay', 'mmc.delay_t', 'delay', 'nan', 'mmc');
end

function TB = add(TB, kind, path, stamp, conv, action, owner)
    TB(end + 1) = struct('kind', kind, 'path', path, 'stamp', stamp, 'conv', conv, ...
                         'action', action, 'owner', owner);
end

function d = table_vs_map(TB, OT)
% Which trimmed variables one side names and the other does not (membership only: the
% conventions, actions and owners are checked on the data, by the oracle).
    m = OT.vars(cellfun(@(v) strcmp(v.role, 'trim'), OT.vars));
    mk = cellfun(@(v) [v.file '/' v.path], m, 'UniformOutput', false);
    tk = arrayfun(@(t) [t.kind '/' t.path], TB, 'UniformOutput', false);
    d = struct('map_only', {setdiff(mk, tk)}, 'oracle_only', {setdiff(tk, mk)});
end

function [res, Tr, Ur] = oracle(T, U, Tr, Ur, d, K, M)
    res = struct('path', d.path, 'owner', d.owner, 'action', d.action, 'n_leaves', 0, ...
                 'n_before', 0, 'n_after', 0, 'n_before_value', 0, 'n_after_value', 0, ...
                 'ok', true, 'why', '', 'marker_ok', true, 'marker_why', '');
    vparts = strsplit(d.path, '.');
    sparts = strsplit(d.stamp, '.');
    if ~has_field_path(U, vparts)
        res.why = 'absent';
        return
    end
    L = K.L.(d.owner);   % the OWNER's start, whether or not it ran (fix 4)
    leaves = expand(U, vparts);
    sib = numel(vparts) == numel(sparts) && isequal(vparts(1:end - 1), sparts(1:end - 1));
    ex = {};
    for i = 1:numel(leaves)
        s = leaves{i};
        if sib
            st = subsref(U, [s(1:end - 1), substruct('.', sparts{end})]);
        else
            st = subsref(U, substruct_path(sparts));
        end
        u = subsref(U, s);
        t = subsref(T, s);
        if iscell(u)
            for c = 1:numel(u)
                [res, e] = check_leaf(res, t{c}, u{c}, st{c}, d, L, K, U);
                ex{end + 1} = e; %#ok<AGROW>
            end
        else
            [res, e] = check_leaf(res, t, u, st, d, L, K, U);
            ex{end + 1} = e; %#ok<AGROW>
        end
        res.n_leaves = res.n_leaves + 1;
        Tr = subsasgn(Tr, s, []);
        Ur = subsasgn(Ur, s, []);
    end
    [res.marker_ok, res.marker_why] = check_marker(M, d, L, K, ex);
end

function [ok, why] = check_marker(M, d, L, K, ex)
% The file's own record of what is not computed, against the oracle - exactly.
    ok = false;
    hit = cellfun(@(v) strcmp(v.path, d.path), M.vars);
    if nnz(hit) ~= 1
        why = sprintf('%d marker entries for %s', nnz(hit), d.path);
        return
    end
    v = M.vars{hit};
    conv = struct('row1', 'row1', 'sec0', 'sec0', 'sec1', 'sec_row1', 'delay', ...
                  'sec_xchan_delay');
    k0 = K.start_sample0.(d.owner);
    checks = {strcmp(M.mode, 'mask_to_electrical_drop_outputs'), 'file mode'; ...
              strcmp(v.mode, 'mask_to_electrical_drop_outputs'), 'mode'; ...
              strcmp(v.owner, d.owner), 'owner'; strcmp(v.action, d.action), 'action'; ...
              strcmp(v.convention, conv.(d.conv)), 'convention'; ...
              strcmp(v.stamp, d.stamp), 'stamp'; ...
              ~isempty(v.convention_meaning), 'convention meaning'; ...
              v.start_sample0 == k0, 'start_sample0'; ...
              abs(v.start_s - k0 / K.fs) < 1e-9, 'start_s'; ...
              v.first_computed_epoch_row == L + 1, 'first_computed_epoch_row'; ...
              v.first_computed_epoch_sample0 == L, 'first_computed_epoch_sample0'; ...
              M.epoch_start_sample0 == K.i0, 'epoch_start_sample0'; ...
              numel(v.leaves) == numel(ex), 'leaf count'};
    for c = 1:size(checks, 1)
        if ~checks{c, 1}
            why = sprintf('%s: marker %s wrong', d.path, checks{c, 2});
            return
        end
    end
    n = 0;
    for i = 1:numel(ex)
        g = v.leaves{i};
        e = ex{i};
        want = e.runs;
        if strcmp(d.action, 'drop'), want = zeros(0, 2); end
        if g.n_entries ~= e.n || g.n_not_computed ~= e.before ...
                || g.n_dropped ~= strcmp(d.action, 'drop') * e.before ...
                || ~isequal(reshape(double(g.not_computed_rows), [], 2), want)
            why = sprintf('%s: marker leaf %d (%s) wrong', d.path, i, g.leaf);
            return
        end
        n = n + e.before;
    end
    if v.n_not_computed ~= n
        why = sprintf('%s: marker n_not_computed %d, oracle %d', d.path, v.n_not_computed, n);
        return
    end
    ok = true;
    why = '';
end

function [res, e] = check_leaf(res, t, u, st, d, L, K, U)
    v = double(st(:));
    switch d.conv
        case 'row1', p = v - 1;
        case 'sec0', p = v * K.fs;
        case 'sec1', p = v * K.fs - 1;
        case 'delay'
            W = double(U.mmc.params.W);
            S = double(U.mmc.params.S);
            p = (v - S + W / 2) * K.fs;
    end
    b = p < L - 1e-6;
    e = struct('n', numel(b), 'before', nnz(b), 'runs', zeros(0, 2));
    if any(b)
        x = diff([false; b; false]);
        e.runs = [find(x == 1), find(x == -1) - 1];
    end
    res.n_before = res.n_before + nnz(b);
    res.n_after = res.n_after + nnz(~b);
    if isempty(u) && isempty(t), return, end
    % entries of the REFERENCE that carry a value (so a wrong cut would be visible); an
    % entry of a list cut by 'drop' is a value whatever it holds (its removal shows)
    if isvector(u) && numel(u) == numel(b)
        has = value_rows(u(:), d.action);
    else
        has = value_rows(u, d.action);
    end
    res.n_before_value = res.n_before_value + nnz(has & b);
    res.n_after_value = res.n_after_value + nnz(has & ~b);
    if isvector(u) && numel(u) == numel(b)
        ub = u(~b); bb = [];
        switch d.action
            case 'drop', ok = bits_equal(t, ub);
            otherwise
                ok = bits_equal(t(~b), ub);
                bb = t(b);
        end
    else
        ub = u(~b, :); bb = [];
        switch d.action
            case 'drop', ok = bits_equal(t, ub);
            otherwise
                ok = bits_equal(t(~b, :), ub);
                bb = t(b, :);
        end
    end
    switch d.action
        case 'nan', ok = ok && all(isnan(bb(:)));
        case 'false', ok = ok && ~any(bb(:));
    end
    if ~ok
        res.ok = false;
        res.why = sprintf('%s mismatch', d.path);
    end
end

function h = value_rows(u, action)
% One flag per row: the row holds a value (not NaN; for a logical series, an event).
    if strcmp(action, 'drop')
        h = true(size(u, 1), 1);
    elseif islogical(u)
        h = any(u, 2);
    elseif isfloat(u)
        h = any(~isnan(u), 2);
    else
        h = true(size(u, 1), 1);
    end
end

function tf = bits_equal(a, b)
    tf = strcmp(class(a), class(b)) && isequal(size(a), size(b));
    if ~tf, return, end
    if isa(a, 'double')
        tf = isequal(typecast(a(:), 'uint64'), typecast(b(:), 'uint64'));
    elseif isa(a, 'single')
        tf = isequal(typecast(a(:), 'uint32'), typecast(b(:), 'uint32'));
    else
        tf = isequal(a, b);
    end
end

function tf = has_field_path(S, parts)
    tf = true;
    v = S;
    for k = 1:numel(parts)
        if ~isstruct(v) || ~isfield(v, parts{k}), tf = false; return, end
        if isempty(v), return, end
        v = v(1).(parts{k});
    end
end

function L = expand(S, parts)
% Every occurrence of the leaf, as a subsref chain (struct arrays indexed element-wise).
    L = {struct('type', {}, 'subs', {})};
    for k = 1:numel(parts) - 1
        nxt = {};
        for i = 1:numel(L)
            v = subsref(S, [L{i}, substruct('.', parts{k})]);
            for e = 1:numel(v)
                nxt{end + 1} = [L{i}, substruct('.', parts{k}, '()', {e})]; %#ok<AGROW>
            end
        end
        L = nxt;
    end
    L = cellfun(@(s) [s, substruct('.', parts{end})], L, 'UniformOutput', false);
end

function s = substruct_path(parts)
    s = struct('type', {}, 'subs', {});
    for k = 1:numel(parts), s = [s, substruct('.', parts{k})]; end %#ok<AGROW>
end

function r = unit_case(K)
% night6_recovery_trim_outputs on hand-built files of four kinds, every stamp convention
% at L - 1, L and L + 1, with owners whose starts differ (hrv vs breathing), a byproduct,
% an unmapped variable, an 'unknown' one and a figure.
    RS = night6_recovery_start(K.starts);
    d = K.dir;
    fs = K.fs;
    Lh = K.L_hrv; Lb = K.L_breathing; Lm = K.L_mmc; Ls = K.L_spikes;
    p3 = @(L) [L - 1; L; L + 1];
    % HRVMeasures (hrv run): sec_row1, sec0, row1
    RR_times = (p3(Lh) + 1) / fs; RR_intervals = [0.11; 0.12; 0.13];
    metrics_t = p3(Lh) / fs; hrv_series = [1; 2; 3]; nRR_used = [4; 5; 6];
    heartlocs = p3(Lh) + 1; hrv = 0.5; mystery = [1 2 3]; t = (0:2)' / fs; %#ok<NASGU>
    save(fullfile(d, 'e2_hrv_HRVMeasures.mat'), 'RR_times', 'RR_intervals', 'metrics_t', ...
         'hrv_series', 'nRR_used', 'heartlocs', 'hrv', 'mystery', 't');
    % HRBR (an hrv-only run: breathRateSeries is a byproduct; and an hrv+breathing run)
    metrics_t = sort([p3(Lh); p3(Lb)]) / fs; heartRateSeries = (1:6)'; %#ok<NASGU>
    breathRateSeries = (11:16)'; br_locs_true = sort([p3(Lh); p3(Lb)]) + 1; %#ok<NASGU>
    RR_implausibleMask = false(3, 1); %#ok<NASGU>
    save(fullfile(d, 'e2_hrv_HRBR.mat'), 'metrics_t', 'heartRateSeries', 'breathRateSeries', ...
         'br_locs_true', 'RR_implausibleMask');
    save(fullfile(d, 'e2_hrv_breathing_HRBR.mat'), 'metrics_t', 'heartRateSeries', ...
         'breathRateSeries', 'br_locs_true', 'RR_implausibleMask');
    % a breathing-only run: heart rate and heartlocs are hrv's byproducts, cut at HRV's start
    heartlocs = sort([p3(Lh); p3(Lb)]) + 1; %#ok<NASGU>
    save(fullfile(d, 'e2_breathing_HRBR.mat'), 'metrics_t', 'heartRateSeries', ...
         'breathRateSeries', 'br_locs_true', 'heartlocs');
    save(fullfile(d, 'e2_breathing_owner_HRBR.mat'), 'metrics_t', 'breathRateSeries');
    % mmc: sec_xchan_delay (true centre = delay_t + W/2 - S) and sec0 on rate_t / t
    mmc = struct();
    mmc.params = struct('W', 10, 'S', 1);
    mmc.delay_t = p3(Lm) / fs - (10 / 2 - 1);
    mmc.delay = [1 1 1; 2 2 2; 3 3 3];
    mmc.t = p3(Lm) / fs;
    mmc.signal = single([1 1 1; 2 2 2; 3 3 3]);
    mmc.rate_t = p3(Lm) / fs;
    mmc.firing = struct('events', true(3, 3), 'rate', ones(3, 3), 'peakAmp', ones(3, 3), ...
                        'avgRate', [1 2 3], 'refractory', 0.05);
    save(fullfile(d, 'e2_mmc_in_mmc.mat'), 'mmc');
    % spikes_v2 (v7.3): a struct array (per channel), co-indexed drop, cells
    spikes = struct('centers', {p3(Ls) + 1, [Ls + 1; Ls + 2]}, ...
                    'times', {p3(Ls) / fs, [Ls; Ls + 1] / fs}, ...
                    'waveforms', {[1 1; 2 2; 3 3], [4 4; 5 5]}, ...
                    'alignedCenters', {p3(Ls) + 1, [Ls + 1; Ls + 2]}, 'nSpikes', {3, 2});
    sigmaWin = struct('centers', {{p3(Ls) + 1, p3(Ls) + 1.5}}, ...
                      'sigma', {{[1; 2; 3], [4; 5; 6]}}, 'windowSec', 5, 'stepFrac', 0.5); %#ok<NASGU>
    save(fullfile(d, 'e2_spikes_v2.mat'), 'spikes', 'sigmaWin', '-v7.3');
    fid = fopen(fullfile(d, 'e2_figure.png'), 'w'); fwrite(fid, 'x'); fclose(fid);
    i0 = K.i0;
    A = @(L) struct('output_rows_before_start', L, 'start_sample0', i0 + L, ...
                    'start_s', (i0 + L) / fs);
    rec = struct('epoch_start_sample0', i0, 'electrical_settle_sample0', i0 + 7, ...
                 'analyses', struct('hrv', A(Lh), 'breathing', A(Lb), 'mmc', A(Lm), ...
                                    'spikes', A(Ls)));
    r = struct();
    r.hrv = night6_recovery_trim_outputs(d, {'e2_hrv_HRVMeasures.mat', 'e2_hrv_HRBR.mat', ...
        'e2_figure.png', '.', '..'}, {'hrv'}, rec, RS.outputTimes, fs);
    r.both = night6_recovery_trim_outputs(d, {'e2_hrv_breathing_HRBR.mat'}, ...
        {'hrv', 'breathing'}, rec, RS.outputTimes, fs);
    r.breathing = night6_recovery_trim_outputs(d, {'e2_breathing_HRBR.mat'}, ...
        {'breathing'}, rec, RS.outputTimes, fs);
    r.out_breathing = load(fullfile(d, 'e2_breathing_HRBR.mat'));
    % trimmed twice: refused; an owner with no start: refused, never cut at another's
    r.twice = attempt(@() night6_recovery_trim_outputs(d, {'e2_breathing_HRBR.mat'}, ...
        {'breathing'}, rec, RS.outputTimes, fs));
    noBr = rec;
    noBr.analyses = rmfield(noBr.analyses, 'breathing');
    r.no_owner = attempt(@() night6_recovery_trim_outputs(d, ...
        {'e2_breathing_owner_HRBR.mat'}, {'hrv'}, noBr, RS.outputTimes, fs));
    r.marker_variable = RS.outputTimes.markerVariable;
    r.mmc = night6_recovery_trim_outputs(d, {'e2_mmc_in_mmc.mat'}, {'mmc'}, rec, ...
        RS.outputTimes, fs);
    r.spikes = night6_recovery_trim_outputs(d, {'e2_spikes_v2.mat'}, {'spikes'}, rec, ...
        RS.outputTimes, fs);
    r.out = struct('hrvm', load(fullfile(d, 'e2_hrv_HRVMeasures.mat')), ...
                   'hrbr', load(fullfile(d, 'e2_hrv_HRBR.mat')), ...
                   'hrbr2', load(fullfile(d, 'e2_hrv_breathing_HRBR.mat')), ...
                   'mmc', load(fullfile(d, 'e2_mmc_in_mmc.mat')), ...
                   'spk', load(fullfile(d, 'e2_spikes_v2.mat')));
    r.out.mmc.mmc.firing.events = double(r.out.mmc.mmc.firing.events);
    r.out.mmc.mmc.signal = double(r.out.mmc.mmc.signal);
    fid = fopen(fullfile(d, 'e2_spikes_v2.mat'), 'r'); h = fread(fid, [1 128], '*char'); fclose(fid);
    r.spikes_still_v73 = contains(h, 'MATLAB 7.3');
    % a mismatched stamp is refused by name, never cut by guess
    metrics_t = (0:3)' / fs; hrv_series = [1; 2; 3]; %#ok<NASGU>
    save(fullfile(d, 'e2_bad_HRVMeasures.mat'), 'metrics_t', 'hrv_series');
    r.shape = attempt(@() night6_recovery_trim_outputs(d, {'e2_bad_HRVMeasures.mat'}, ...
        {'hrv'}, rec, RS.outputTimes, fs));
end

function r = sources_case(K)
% night6_recovery_start against the code MATLAB resolves: a copy of one cited function
% whose bytes differ, first on the path, is refused by name; a byte-identical copy is not.
    src = which(K.name);
    dst = fullfile(K.dir, K.name);
    r = struct('src', src);
    copyfile(src, dst);
    addpath(K.dir, '-begin');
    cleanup = onCleanup(@() rmpath(K.dir));
    r.same_which = which(K.name);
    r.same = attempt(@() night6_recovery_start(K.starts));
    fid = fopen(dst, 'a');
    fwrite(fid, sprintf('\n%% a changed copy\n'));
    fclose(fid);
    clear(K.name(1:end - 2));
    r.changed_which = which(K.name);
    r.changed = attempt(@() night6_recovery_start(K.starts));
    delete(cleanup);
    r.after_which = which(K.name);
    r.after = attempt(@() night6_recovery_start(K.starts));
end

function r = batch_case(K)
% night6_batch refuses at batch start, by name, before any recording is loaded.
    r = struct();
    for c = as_cells(K.lists)
        r.(c{1}.name) = attempt(@() night6_batch(c{1}.file, 0, 'DryRun', true));
    end
end
