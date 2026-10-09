function check_recovery_start(caseFile, outFile)
% CHECK_RECOVERY_START  Harness for tests/test_night6_recovery_start.py (RULING 2026-10-08 (k) 2,
% RULING 2026-10-09 item 6).
%
%   plans    each case plans ONE epoch from a mask file written by the real
%            emit.handoff.write_mask_file, twice: without 'Recovery' (the untrimmed
%            baseline) and with the case's starts file. For both it reports, per run and
%            per input column, the NaN rows of night6_consumer_input (1-based inclusive
%            runs), plus plan.recoveryStart - or the error identifier and message.
%   readers  night6_recovery_start on each file: 'ok' or the error identifier.
%   run      night6_run_recording (DryRun) on a store mask folder: without RecoveryStarts,
%            with one, without / with an unknown / with the withdrawn (A) RecoveryTrimMode,
%            and the resume rule (a complete record made with another starts file, or
%            under (A), reruns; one made with this file and mode is skipped).
%   trim     mode (B) end to end with her real functions, against a reference run cut at
%            the electrical settling on the same masked input, through an oracle with its
%            own tables of conventions, CLASSES and CUTS, windowed variables and
%            recomputed averages (trim_case, oracle_table, oracle_fractions,
%            oracle_recomputed), and the marker in every file (RULING 2026-10-09 item 6).
%   unit     night6_recovery_trim_outputs on hand-built files: every class and convention at
%            L-1/L/L+1 of its own cut, exact valid fractions, averages, NaN events, refusals.
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
    args = [args0, {'RecoveryTrimMode', 'mask_to_electrical_drop_outputs'}];
    r = struct();
    r.without = attempt(@() night6_run_recording(K.mask_folder, 'OutRoot', K.out_a, args{:}));
    r.no_mode = attempt(@() night6_run_recording(K.mask_folder, 'OutRoot', K.out_a, ...
        'RecoveryStarts', K.starts_file, args0{:}));
    r.bad_mode = attempt(@() night6_run_recording(K.mask_folder, 'OutRoot', K.out_a, ...
        'RecoveryStarts', K.starts_file, 'RecoveryTrimMode', 'mask_everything', args0{:}));
    r.mode_a = attempt(@() night6_run_recording(K.mask_folder, 'OutRoot', K.out_a, ...
        'RecoveryStarts', K.starts_file, 'RecoveryTrimMode', 'mask_to_own_start', args0{:}));
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
    % a complete record made under the withdrawn mode (A), same starts file: rerun under (B)
    Rc = jsondecode(fileread(recFile));
    Rc.status = 'complete';
    Rc.recovery_trim_mode = 'mask_to_own_start';
    write(recFile, Rc);
    R4 = night6_run_recording(K.mask_folder, 'OutRoot', K.out_b, 'RecoveryStarts', ...
                              K.other_starts_file, args{:});
    r.resume_old_mode = R4{1}.status;
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
% mode (B), per output variable (RULING 2026-10-09 item 6)

function r = trim_case(K)
% Mode (B) end to end, with her real functions: the same epoch run twice on the same
% masked input (every input masked to the electrical settling) - once with the case's
% per-variable cuts, once as the reference with EVERY cut at the electrical settling.
% Every output file is compared through an ORACLE that does not read the map: its own
% table (oracle_table, transcribed from her code with file:line) says which variables
% are time-stamped, by what, in which convention, with which action, owner, CLASS and
% CUT; each entry is cut at its own cut (K.cuts). Its own tables of the windowed
% variables (oracle_fractions) and of the recomputed averages (oracle_recomputed) check
% the valid fractions and the averages. The marker each trimmed file carries is checked
% against the same oracle, exactly.
    args = {'GemsRoot', K.gems_root, 'Units', 'uV', 'CodeCommit', 'test'};
    mode = 'mask_to_electrical_drop_outputs';
    rng(0, 'twister');
    Rd = night6_run_recording(K.mask_folder, 'OutRoot', K.out_drop, 'RecoveryStarts', ...
                              K.starts_drop, 'RecoveryTrimMode', mode, args{:});
    rng(0, 'twister');
    Rr = night6_run_recording(K.mask_folder, 'OutRoot', K.out_ref, 'RecoveryStarts', ...
                              K.starts_ref, 'RecoveryTrimMode', mode, args{:});
    r = struct('record_drop', jsonencode(Rd{1}), 'record_ref', jsonencode(Rr{1}), ...
               'files', {{}}, 'vars', {{}}, 'fractions', {{}}, 'recomputed', {{}});
    RS = night6_recovery_start(K.starts_drop);
    OT = RS.outputTimes;
    TB = oracle_table();
    FR = oracle_fractions();
    RC = oracle_recomputed();
    r.table_vs_map = table_vs_map(TB, OT);
    r.recomputed_vs_map = recomputed_vs_map(RC, OT);
    r.marker_variable = OT.markerVariable;
    cut = containers.Map();
    for c = as_cells(K.cuts), cut(c{1}.cut) = c{1}; end
    dD = fullfile(K.out_drop, K.epoch_rel);
    dR = fullfile(K.out_ref, K.epoch_rel);
    files = dir(fullfile(dD, '*.mat'));
    for k = 1:numel(files)
        name = files(k).name;
        T = load(fullfile(dD, name));
        U = load(fullfile(dR, name));
        hit = cellfun(@(f) ~isempty(regexp(name, f.pattern, 'once')), OT.files);
        fr = struct('file', name, 'kind', '', 'rest_equal', false, 'has_marker', false, ...
                    'marker_extra', {{}}, 'added_ok', false);
        if ~any(hit)
            fr.rest_equal = isequaln(T, U);
            r.files{end + 1} = fr;
            continue
        end
        kind = OT.files{hit}.kind;
        fr.kind = kind;
        fr.has_marker = isfield(T, OT.markerVariable) && isfield(U, OT.markerVariable);
        M = struct('vars', {{}}, 'added_variables', {{}}, 'recomputed', {{}});
        MU = M;
        if fr.has_marker, M = T.(OT.markerVariable); MU = U.(OT.markerVariable); end
        mine = TB(strcmp({TB.kind}, kind));
        Tr = T; Ur = U;
        seen = {};
        for j = 1:numel(mine)
            [res, Tr, Ur] = oracle(T, U, Tr, Ur, mine(j), K, M, cut);
            res.file = name;
            r.vars{end + 1} = res;
            if ~strcmp(res.why, 'absent'), seen{end + 1} = mine(j).path; end %#ok<AGROW>
        end
        mpaths = cellfun(@(v) v.path, M.vars, 'UniformOutput', false);
        fr.marker_extra = setdiff(mpaths, seen);   % the marker names nothing it did not cut
        added = {};
        for f = FR(strcmp({FR.kind}, kind))
            res = check_fraction(T, U, M, f, K);
            res.file = name;
            r.fractions{end + 1} = res;
            if strcmp(f.how, 'sibling') && ~strcmp(res.why, 'absent')
                added{end + 1} = [f.path '_validFraction']; %#ok<AGROW>
            end
        end
        fr.added_ok = isequal(sort(as_cells(M.added_variables)), sort(added)) ...
                      && isequal(sort(as_cells(MU.added_variables)), sort(added));
        for c = RC(strcmp({RC.kind}, kind))
            [res, Tr, Ur] = check_recomputed(T, U, Tr, Ur, M, MU, c, K);
            res.file = name;
            r.recomputed{end + 1} = res;
        end
        if strcmp(kind, 'mmc')   % her qc.srcFile names the input file: the out folder
            Tr.mmc.qc.srcFile = ''; Ur.mmc.qc.srcFile = '';
        end
        if fr.has_marker
            Tr = rmfield(Tr, OT.markerVariable);
            Ur = rmfield(Ur, OT.markerVariable);
        end
        fr.rest_equal = isequaln(Tr, Ur);   % everything else - the siblings too - untouched
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
% Owner = the analysis the variable belongs to (an HR call writes both hrv's and
% breathing's outputs). Action: nan (value NaN, stamp kept), drop (removed with its
% co-indexed lists), nan_events (logical event series made double, NaN = not computed).
% CLASS (RULING 2026-10-09 item 6): I = valid_only - computed over valid samples or
% events only; II = filled_or_filtered - computed on filled-in or filtered data. CUT =
% owner.output.class: the cut point each variable is kept from.
    I = 'valid_only'; II = 'filled_or_filtered';
    TB = struct('kind', {}, 'path', {}, 'stamp', {}, 'conv', {}, 'action', {}, 'owner', {}, ...
                'cls', {}, 'cut', {});
    % --- spikes: process_dataset_v2 (night6_run_recording save_spikes_v2: her D fields)
    % step2_noise_sigma.m:89-98 window i0 = (w-1)*step+1 (1-based), cWin(w) = (i0+i1)/2;
    % :95 the window's VALID samples only (>= 20 %): class I
    TB = add(TB, 'spikes_v2', 'sigmaWin.sigma', 'sigmaWin.centers', 'row1', 'nan', 'spikes', ...
             I, 'spikes.sigma_windows.valid_only');
    % step3_detect.m:86-90 centers = locs (findpeaks on D.filtered: step1_bandpass.m:53-57
    % fills and filtfilts): class II
    for v = {'centers', 'times', 'peakAmp_uv', 'threshAtSpike_uv', 'artifactMask'}
        TB = add(TB, 'spikes_v2', ['spikes.' v{1}], 'spikes.centers', 'row1', 'drop', 'spikes', ...
                 II, 'spikes.spike_times.filled_or_filtered');
    end
    % step4_waveforms.m:114-118 alignedCenters (rows), alignedTimes = (aligned-1)/fs: II
    for v = {'waveforms', 'alignedCenters', 'alignedTimes', 'Vpp_uv', 'width_ms'}
        TB = add(TB, 'spikes_v2', ['spikes.' v{1}], 'spikes.alignedCenters', 'row1', 'drop', ...
                 'spikes', II, 'spikes.spike_waveforms.filled_or_filtered');
    end
    % step3b_envelope.m:95-107 bins, t_c(b) = ((i0+i1)/2 - 1)/fs; :99 RMS over valid samples: I
    for v = {'rms_uv', 'sigmaFloor_uv', 'excess_uv', 'validFrac'}
        TB = add(TB, 'spikes_v2', ['envelope.' v{1}], 'envelope.t', 'sec0', 'nan', 'spikes', ...
                 I, 'spikes.envelope.valid_only');
    end
    % step6_spike_report.m:62, firing_rate :142-151 t(b) = ((i0+i1)/2-1)/fs; :149 valid s: I
    for v = {'fr_hz', 'fr_validFrac'}
        TB = add(TB, 'spikes_v2', ['metrics.' v{1}], 'metrics.fr_t', 'sec0', 'nan', 'spikes', ...
                 I, 'spikes.firing_rate.valid_only');
    end
    % step6_spike_report.m:94, rolling_cv2 :262-269 on gap-clean ISIs (:49): I
    TB = add(TB, 'spikes_v2', 'metrics.cv2_roll', 'metrics.cv2_t', 'sec0', 'nan', 'spikes', ...
             I, 'spikes.cv2.valid_only');
    % step6_spike_report.m:249-252 onsets/offsets = st(i0)/st(i1) of gap-clean ISIs, st the
    % aligned times: I, reaching back as the aligned spikes do
    for v = {'onsets', 'offsets'}
        TB = add(TB, 'spikes_v2', ['metrics.burst.' v{1}], 'metrics.burst.onsets', 'sec0', ...
                 'drop', 'spikes', I, 'spikes.spike_waveforms.valid_only');
    end
    % --- HR_BR_HRVAnalysis_beats: _HRBR.mat (save :663-674), _HRVMeasures.mat (:685-694)
    % :282 yFilt = filtfilt over the linear fill (:262); :303 heartBeatSeries: II
    TB = add(TB, 'HRBR', 'heartBeatSeries', 't', 'sec0', 'nan', 'hrv', II, ...
             'hrv.heart_band_trace.filled_or_filtered');
    % :292-299 heartlocs: stored beats, those at invalid samples rejected: I
    for f = {'HRBR', 'HRVMeasures'}
        TB = add(TB, f{1}, 'heartlocs', 'heartlocs', 'row1', 'drop', 'hrv', I, ...
                 'hrv.beats.valid_only');
    end
    % :792 metrics_t; :844-849 HR over the longest clean stretch (I, heart_rate); :893-896
    % counts over valid samples (I, count/HRV window); :866 breath rate (breathing)
    TB = add(TB, 'HRBR', 'heartRateSeries', 'metrics_t', 'sec0', 'nan', 'hrv', I, ...
             'hrv.heart_rate.valid_only');
    for v = {'heartCountSeries', 'heartCountValidSec', 'heartCountRateSeries'}
        TB = add(TB, 'HRBR', v{1}, 'metrics_t', 'sec0', 'nan', 'hrv', I, ...
                 'hrv.count_hrv.valid_only');
    end
    TB = add(TB, 'HRBR', 'breathRateSeries', 'metrics_t', 'sec0', 'nan', 'breathing', I, ...
             'breathing.breath_rate.valid_only');
    % :388 br_locs_true = heartlocs(br_locs), :391 those at invalid samples rejected: I
    TB = add(TB, 'HRBR', 'br_locs_true', 'br_locs_true', 'row1', 'drop', 'breathing', I, ...
             'breathing.breath_troughs.valid_only');
    % :310, computeValidRRIntervals :983-1000: RR_times = s1/fs; no invalid sample between: I
    for v = {'RR_intervals', 'RR_times'}
        TB = add(TB, 'HRVMeasures', v{1}, 'RR_times', 'sec1', 'drop', 'hrv', I, ...
                 'hrv.beats.valid_only');
    end
    % :901-917 windowed HRV over valid RR intervals (:902 >= minRR): I
    for v = {'hrv_series', 'rmssd_series', 'pnn5_series', 'sd1_series', 'sd2_series', ...
             'nRR_used'}
        TB = add(TB, 'HRVMeasures', v{1}, 'metrics_t', 'sec0', 'nan', 'hrv', I, ...
                 'hrv.count_hrv.valid_only');
    end
    % :920-929 sample entropy over valid RR intervals (fixed 60 s): I
    TB = add(TB, 'HRVMeasures', 'sampEn_series', 'metrics_t', 'sec0', 'nan', 'hrv', I, ...
             'hrv.sampen.valid_only');
    % --- slowWaveAnalysis_new (save :340-345; one file per kept channel): fillmissing
    % (:126) before the low-pass (:133): every output II
    TB = add(TB, 'slowWaves', 'slowWaveTimeSeries', 't', 'sec0', 'nan', 'slow_wave', II, ...
             'slow_wave.sw_trace.filled_or_filtered');
    TB = add(TB, 'slowWaves', 'slowWaveRateSeries', 'slowWaveRateTime', 'sec0', 'nan', ...
             'slow_wave', II, 'slow_wave.sw_rate.filled_or_filtered');
    TB = add(TB, 'slowWaves', 'slowWavePeakLocs', 'slowWavePeakLocs', 'row1', 'drop', ...
             'slow_wave', II, 'slow_wave.sw_trace.filled_or_filtered');
    % --- extract_mmc (save :170; struct :151-166)
    % :104-106 filtfilt over the fill; :153 signal = single(cond): II
    TB = add(TB, 'mmc', 'mmc.signal', 'mmc.t', 'sec0', 'nan', 'mmc', II, ...
             'mmc.mmc_signal.filled_or_filtered');
    for lvl = {'firing', 'burst'}
        % :155-157 events = ev_bool, detected on the filtered signal: II, NaN when not computed
        TB = add(TB, 'mmc', ['mmc.' lvl{1} '.events'], 'mmc.t', 'sec0', 'nan_events', 'mmc', ...
                 II, 'mmc.mmc_events.filled_or_filtered');
        % :284-293 rate / peak amplitude over valid samples, >= 0.5 valid (:290): I
        for v = {'rate', 'peakAmp'}
            TB = add(TB, 'mmc', ['mmc.' lvl{1} '.' v{1}], 'mmc.rate_t', 'sec0', 'nan', 'mmc', ...
                     I, 'mmc.mmc_rate.valid_only');
        end
    end
    % :118 xchan_delay on the firing rate; :270 invalid rows mean-filled: II
    TB = add(TB, 'mmc', 'mmc.delay', 'mmc.delay_t', 'delay', 'nan', 'mmc', II, ...
             'mmc.mmc_delay.filled_or_filtered');
end

function TB = add(TB, kind, path, stamp, conv, action, owner, cls, cut)
    TB(end + 1) = struct('kind', kind, 'path', path, 'stamp', stamp, 'conv', conv, ...
                         'action', action, 'owner', owner, 'cls', cls, 'cut', cut);
end

function FR = oracle_fractions()
% Every windowed output (each value carries its valid fraction) and where it lives:
% 'sibling' = <path>_validFraction added by Night 6, else her own variable.
    FR = struct('kind', {}, 'path', {}, 'how', {});
    FR(end + 1) = struct('kind', 'spikes_v2', 'path', 'sigmaWin.sigma', 'how', 'sibling');
    for v = {'rms_uv', 'sigmaFloor_uv', 'excess_uv', 'validFrac'}
        FR(end + 1) = struct('kind', 'spikes_v2', 'path', ['envelope.' v{1}], ...
                             'how', 'envelope.validFrac'); %#ok<AGROW>
    end
    for v = {'fr_hz', 'fr_validFrac'}
        FR(end + 1) = struct('kind', 'spikes_v2', 'path', ['metrics.' v{1}], ...
                             'how', 'metrics.fr_validFrac'); %#ok<AGROW>
    end
    FR(end + 1) = struct('kind', 'spikes_v2', 'path', 'metrics.cv2_roll', 'how', 'sibling');
    for v = {'heartRateSeries', 'heartCountSeries', 'heartCountValidSec', ...
             'heartCountRateSeries', 'breathRateSeries'}
        FR(end + 1) = struct('kind', 'HRBR', 'path', v{1}, 'how', 'sibling'); %#ok<AGROW>
    end
    for v = {'hrv_series', 'rmssd_series', 'pnn5_series', 'sd1_series', 'sd2_series', ...
             'sampEn_series', 'nRR_used'}
        FR(end + 1) = struct('kind', 'HRVMeasures', 'path', v{1}, 'how', 'sibling'); %#ok<AGROW>
    end
    FR(end + 1) = struct('kind', 'slowWaves', 'path', 'slowWaveRateSeries', 'how', 'sibling');
    for lvl = {'firing', 'burst'}
        for v = {'rate', 'peakAmp'}
            FR(end + 1) = struct('kind', 'mmc', 'path', ['mmc.' lvl{1} '.' v{1}], ...
                                 'how', 'sibling'); %#ok<AGROW>
        end
    end
    FR(end + 1) = struct('kind', 'mmc', 'path', 'mmc.delay', 'how', 'sibling');
end

function RC = oracle_recomputed()
% Every whole-epoch average or count of a trimmed series, and her expression for it.
    RC = struct('kind', {}, 'path', {}, 'of', {}, 'how', {});
    for p = {{'avgHeartRate', 'heartRateSeries'}, {'avgBreathRate', 'breathRateSeries'}, ...
             {'avgHeartCount', 'heartCountSeries'}, ...
             {'avgHeartCountRate', 'heartCountRateSeries'}}     % HR_BR :436-442
        RC(end + 1) = struct('kind', 'HRBR', 'path', p{1}{1}, 'of', p{1}{2}, 'how', 'mean'); %#ok<AGROW>
    end
    RC(end + 1) = struct('kind', 'slowWaves', 'path', 'avgSlowWave', ...
                         'of', 'slowWaveRateSeries', 'how', 'mean');           % SW :283
    for lvl = {'firing', 'burst'}                                              % mmc :295
        RC(end + 1) = struct('kind', 'mmc', 'path', ['mmc.' lvl{1} '.avgRate'], ...
                             'of', ['mmc.' lvl{1} '.events'], 'how', 'events'); %#ok<AGROW>
    end
    RC(end + 1) = struct('kind', 'spikes_v2', 'path', 'spikes.nSpikes', ...
                         'of', 'spikes.alignedCenters', 'how', 'count');       % step4 :122
    RC(end + 1) = struct('kind', 'spikes_v2', 'path', 'metrics.nSpikes', ...
                         'of', 'spikes.alignedTimes', 'how', 'count');         % step6 :55
    RC(end + 1) = struct('kind', 'spikes_v2', 'path', 'envelope.meanRMS_uv', ...
                         'of', 'envelope.rms_uv', 'how', 'well');              % 3b :110-111
    RC(end + 1) = struct('kind', 'spikes_v2', 'path', 'envelope.meanExcess_uv', ...
                         'of', 'envelope.excess_uv', 'how', 'well');           % 3b :112
end

function d = table_vs_map(TB, OT)
% Which trimmed variables one side names and the other does not (membership only: the
% conventions, actions, owners, classes and cuts are checked on the data, by the oracle).
    m = OT.vars(cellfun(@(v) strcmp(v.role, 'trim'), OT.vars));
    mk = cellfun(@(v) [v.file '/' v.path], m, 'UniformOutput', false);
    tk = arrayfun(@(t) [t.kind '/' t.path], TB, 'UniformOutput', false);
    d = struct('map_only', {setdiff(mk, tk)}, 'oracle_only', {setdiff(tk, mk)});
end

function d = recomputed_vs_map(RC, OT)
    m = OT.vars(cellfun(@(v) strcmp(v.role, 'recomputed'), OT.vars));
    mk = cellfun(@(v) [v.file '/' v.path], m, 'UniformOutput', false);
    tk = arrayfun(@(t) [t.kind '/' t.path], RC, 'UniformOutput', false);
    d = struct('map_only', {setdiff(mk, tk)}, 'oracle_only', {setdiff(tk, mk)});
end

function [res, Tr, Ur] = oracle(T, U, Tr, Ur, d, K, M, cut)
    res = struct('path', d.path, 'owner', d.owner, 'action', d.action, 'cls', d.cls, ...
                 'cut', d.cut, 'n_leaves', 0, 'n_before', 0, 'n_after', 0, ...
                 'n_before_value', 0, 'n_after_value', 0, 'ok', true, 'why', '', ...
                 'marker_ok', true, 'marker_why', '');
    vparts = strsplit(d.path, '.');
    sparts = strsplit(d.stamp, '.');
    if ~has_field_path(U, vparts)
        res.why = 'absent';
        return
    end
    C = cut(d.cut);   % ITS OWN cut, whichever analysis ran
    L = double(C.L);
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
    [res.marker_ok, res.marker_why] = check_marker(M, d, L, C, K, ex);
end

function [ok, why] = check_marker(M, d, L, C, K, ex)
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
    k0 = double(C.sample0);
    checks = {strcmp(M.mode, 'mask_to_electrical_drop_outputs'), 'file mode'; ...
              contains(M.ruling, 'RULING 2026-10-09 item 6'), 'ruling'; ...
              strcmp(v.mode, 'mask_to_electrical_drop_outputs'), 'mode'; ...
              strcmp(v.owner, d.owner), 'owner'; strcmp(v.action, d.action), 'action'; ...
              strcmp(v.trim_class, d.cls), 'trim class'; strcmp(v.cut, d.cut), 'cut'; ...
              ~isempty(v.trim_class_meaning), 'class meaning'; ...
              strcmp(v.convention, conv.(d.conv)), 'convention'; ...
              strcmp(v.stamp, d.stamp), 'stamp'; ...
              ~isempty(v.convention_meaning), 'convention meaning'; ...
              v.start_sample0 == k0, 'start_sample0'; ...
              abs(v.start_s - k0 / K.fs) < 1e-9, 'start_s'; ...
              v.first_computed_epoch_row == L + 1, 'first_computed_epoch_row'; ...
              v.first_computed_epoch_sample0 == L, 'first_computed_epoch_sample0'; ...
              M.epoch_start_sample0 == K.i0, 'epoch_start_sample0'; ...
              strcmp(d.cls, 'filled_or_filtered') || ~isempty(v.edge_rule.source), 'edge rule'; ...
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
        case 'nan_events'
            % NOT COMPUTED is NaN, never "no event": the series is double, and the rows
            % after the cut hold exactly her events (1) and non-events (0)
            ok = ok && strcmp(class(t), 'double') && all(isnan(bb(:))) ...
                 && all(ismember(ub(:), [0 1]));
    end
    if ~ok
        res.ok = false;
        res.why = sprintf('%s mismatch', d.path);
    end
end

function h = value_rows(u, action)
% One flag per row: the row holds a value (not NaN; for an event series, an event).
    if strcmp(action, 'drop')
        h = true(size(u, 1), 1);
    elseif strcmp(action, 'nan_events')
        h = any(u == 1, 2);
    elseif islogical(u)
        h = any(u, 2);
    elseif isfloat(u)
        h = any(~isnan(u), 2);
    else
        h = true(size(u, 1), 1);
    end
end

function res = check_fraction(T, U, M, f, K)
% Each windowed value's valid fraction: Night 6's sibling (the same in both runs - it is
% computed from the masked input, never from the cut) or her own variable, as the marker
% says; in [0, 1]; and consistent with her own rules where she has one.
    res = struct('path', f.path, 'how', f.how, 'ok', false, 'why', '', 'n_values', 0, ...
                 'n_partial', 0, 'n_rule', 0);
    vparts = strsplit(f.path, '.');
    if ~has_field_path(U, vparts)
        res.why = 'absent';
        return
    end
    hit = cellfun(@(v) strcmp(v.path, f.path), M.vars);
    if nnz(hit) ~= 1 || ~isstruct(M.vars{hit}.valid_fraction)
        res.why = 'no valid_fraction in the marker';
        return
    end
    mv = M.vars{hit}.valid_fraction;
    if strcmp(f.how, 'sibling')
        want = [f.path '_validFraction'];
        if ~strcmp(mv.variable, want) || ~mv.added
            res.why = sprintf('marker names %s', mv.variable);
            return
        end
        sparts = [vparts(1:end - 1), {[vparts{end} '_validFraction']}];
    else
        if ~strcmp(mv.variable, f.how) || mv.added || ~strcmp(mv.kind, 'her')
            res.why = sprintf('marker names %s for her %s', mv.variable, f.how);
            return
        end
        sparts = strsplit(f.how, '.');
        if has_field_path(T, [vparts(1:end - 1), {[vparts{end} '_validFraction']}])
            res.why = 'a sibling was added beside her own fraction';
            return
        end
    end
    leaves = expand(U, vparts);
    for i = 1:numel(leaves)
        s = leaves{i};
        fs_ = [s(1:end - 1), substruct('.', sparts{end})];
        if numel(sparts) ~= numel(vparts)
            fs_ = substruct_path(sparts);
        end
        u = subsref(U, s);
        fT = subsref(T, fs_);
        fU = subsref(U, fs_);
        if ~iscell(u), u = {u}; fT = {fT}; fU = {fU}; end
        for c = 1:numel(u)
            a = fT{c}; b = fU{c};
            if strcmp(f.how, 'sibling') && ~(isequal(size(a), size(u{c})) && bits_equal(a, b))
                res.why = sprintf('%s leaf %d: sibling differs from the reference', f.path, i);
                return
            end
            if any(a(:) < 0 | a(:) > 1)
                res.why = sprintf('%s leaf %d: a fraction outside [0, 1]', f.path, i);
                return
            end
            res.n_values = res.n_values + numel(a);
            res.n_partial = res.n_partial + nnz(a > 0 & a < 1);
            [ok, n] = her_rule(f.path, u{c}, b, U, K);
            res.n_rule = res.n_rule + n;
            if ~ok
                res.why = sprintf('%s leaf %d: disagrees with her rule', f.path, i);
                return
            end
        end
    end
    res.ok = true;
end

function [ok, n] = her_rule(path, u, f, U, K)
% Where her own code gates on validity, the fraction must agree with what she computed.
    ok = true; n = 0;
    switch path
        case 'heartCountRateSeries'   % HR_BR :895: rate iff >= 0.5 of the window valid
            evald = isfinite(U.heartCountSeries(:));     % windows she evaluated
            n = nnz(evald);
            ok = isequal(isfinite(u(evald)), f(evald) >= 0.5);
        case 'heartCountValidSec'     % :894: valid seconds / valid fraction = the window
            ev = isfinite(u(:)) & f(:) > 0;
            n = nnz(ev);
            ok = all(abs(u(ev) ./ f(ev) - double(U.winSec)) <= 2 / K.fs);
        case {'mmc.firing.rate', 'mmc.burst.rate'}   % extract_mmc :290: >= 0.5 of W valid
            W = double(U.mmc.params.W);
            tol = 2 / (W * K.fs);
            n = numel(u);
            ok = all(f(isfinite(u)) >= 0.5 - tol) && all(~isfinite(u(f < 0.5 - tol)));
    end
end

function [res, Tr, Ur] = check_recomputed(T, U, Tr, Ur, M, MU, c, K)
% A recomputed average equals her expression over the KEPT values; the marker keeps her
% original (the same in both runs: one call), labelled.
    res = struct('path', c.path, 'ok', false, 'why', '', 'n_leaves', 0, 'n_changed', 0);
    if ~has_field_path(U, strsplit(c.path, '.'))
        res.why = 'absent';
        return
    end
    hit = cellfun(@(e) strcmp(e.path, c.path), as_cells(M.recomputed));
    hitU = cellfun(@(e) strcmp(e.path, c.path), as_cells(MU.recomputed));
    if nnz(hit) ~= 1 || nnz(hitU) ~= 1
        res.why = 'not in the marker exactly once';
        return
    end
    e = M.recomputed{hit};
    eU = MU.recomputed{hitU};
    if ~contains(e.label, 'RECOMPUTED')
        res.why = 'not labelled as recomputed';
        return
    end
    tparts = strsplit(c.path, '.');
    leaves = expand(U, tparts);
    if numel(e.leaves) ~= numel(leaves)
        res.why = 'marker leaf count';
        return
    end
    for i = 1:numel(leaves)
        s = leaves{i};
        k = 1;
        if numel(s) >= 2 && strcmp(s(2).type, '()'), k = s(2).subs{1}; end
        got = subsref(T, s);
        x = elem(T, c.of, k);
        switch c.how
            case 'mean', want = mean(x, 1, 'omitnan');
            case 'count', want = numel(x);
            case 'well'
                rms = elem(T, 'envelope.rms_uv', k);
                vf = elem(T, 'envelope.validFrac', k);
                want = mean(x(vf >= 0.5 & isfinite(rms)), 'omitnan');
            case 'events'
                sig = T.mmc.signal;   % trimmed at its own, earlier, cut: NaN only there
                kept = ~isnan(x);
                want = zeros(1, size(x, 2));
                for ch = 1:size(x, 2)
                    want(ch) = nnz(x(kept(:, ch), ch) == 1) / ...
                               max(nnz(kept(:, ch) & ~isnan(sig(:, ch))) / K.fs, eps);
                end
        end
        g = e.leaves{i};
        if ~(isequaln(double(got), double(want)) && isequaln(g.recomputed, got) ...
                && isequaln(g.original, eU.leaves{i}.original))
            res.why = sprintf('%s leaf %d: recomputed %s, oracle %s', c.path, i, ...
                              mat2str(double(got)), mat2str(double(want)));
            return
        end
        res.n_changed = res.n_changed + ~isequaln(g.original, got);
        res.n_leaves = res.n_leaves + 1;
        Tr = subsasgn(Tr, s, []);
        Ur = subsasgn(Ur, s, []);
    end
    res.ok = true;
end

function v = elem(S, path, k)
% A path read at struct-array element k (scalar structs as they are).
    v = S;
    for p = strsplit(path, '.')
        if numel(v) > 1, v = v(k); end
        v = v.(p{1});
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
% night6_recovery_trim_outputs on hand-built files of every kind, each variable at
% L - 1, L and L + 1 of ITS OWN cut (every cut different): every class and convention,
% byproducts in a breathing-only run, struct arrays, cells and co-indexed lists, v7.3
% kept, an unmapped and an 'unknown' variable and a figure listed untrimmed; the valid
% fractions of every computed kind on known masks (the Python test computes them
% independently); the recomputed averages and counts; the mmc events NaN; and the
% refusals (trimmed twice, no cut, no class, an unknown class, a stamp that does not fit,
% a sibling that already exists).
    RS = night6_recovery_start(K.starts);
    OT = RS.outputTimes;
    d = K.dir; fs = K.fs; i0 = K.i0; N = K.n;
    L = containers.Map();
    for c = as_cells(K.cut_L), L(c{1}.cut) = double(c{1}.L); end
    p3 = @(x) [x - 1; x; x + 1];
    invalid = false(N, 1);
    for j = 1:size(K.invalid, 1), invalid(K.invalid(j, 1):K.invalid(j, 2)) = true; end
    Lb = L('hrv.beats.valid_only'); Lc = L('hrv.count_hrv.valid_only');
    Lh = L('hrv.heart_rate.valid_only'); Lt = L('hrv.heart_band_trace.filled_or_filtered');
    Lbr = L('breathing.breath_rate.valid_only'); Ltr = L('breathing.breath_troughs.valid_only');
    % HRVMeasures (an hrv run): class (i) RR (sec_row1), heartlocs (row1), hrv_series (sec0)
    RR_times = (p3(Lb) + 1) / fs; RR_intervals = [0.11; 0.12; 0.13]; %#ok<NASGU>
    heartlocs = p3(Lb) + 1;
    metrics_t = p3(Lc) / fs; hrv_series = [1; 2; 3]; nRR_used = [4; 5; 6]; %#ok<NASGU>
    hrv = 0.5; mystery = [1 2 3]; invalidMask = invalid; winSec = K.win_w / fs; %#ok<NASGU>
    save(fullfile(d, 'e2_hrv_HRVMeasures.mat'), 'RR_times', 'RR_intervals', 'heartlocs', ...
         'metrics_t', 'hrv_series', 'nRR_used', 'hrv', 'mystery', 'invalidMask', 'winSec');
    % HRBR: heart rate, counts and breath rate on one axis, each at its own cut; the class
    % (ii) trace at its own; the breath troughs; the averages
    metrics_t = sort([p3(Lh); p3(Lbr); p3(Lc)]) / fs;
    heartRateSeries = (1:9)'; breathRateSeries = (11:19)'; heartCountSeries = (21:29)'; %#ok<NASGU>
    heartCountRateSeries = (31:39)'; heartCountValidSec = (41:49)' / 100; %#ok<NASGU>
    t = p3(Lt) / fs; heartBeatSeries = [1; 2; 3]; br_locs_true = p3(Ltr) + 1; %#ok<NASGU>
    avgHeartRate = 99; avgBreathRate = 98; avgHeartCount = 97; avgHeartCountRate = 96; %#ok<NASGU>
    hrBrWinSec = K.hrbr_w / fs; RR_implausibleMask = false(3, 1); %#ok<NASGU>
    v = {'metrics_t', 'heartRateSeries', 'breathRateSeries', 'heartCountSeries', ...
         'heartCountRateSeries', 'heartCountValidSec', 't', 'heartBeatSeries', ...
         'br_locs_true', 'heartlocs', 'avgHeartRate', 'avgBreathRate', 'avgHeartCount', ...
         'avgHeartCountRate', 'hrBrWinSec', 'winSec', 'invalidMask', 'RR_implausibleMask'};
    save(fullfile(d, 'e2_hrv_HRBR.mat'), v{:});
    save(fullfile(d, 'e2_breathing_HRBR.mat'), v{:});   % a breathing-only run
    save(fullfile(d, 'e2_nocut_HRBR.mat'), v{:});
    save(fullfile(d, 'e2_noclass_HRBR.mat'), v{:});
    heartRateSeries_validFraction = ones(9, 1); %#ok<NASGU>
    save(fullfile(d, 'e2_collide_HRBR.mat'), v{:}, 'heartRateSeries_validFraction');
    % mmc, full rate: signal and events at their own cuts, rate (6 rows) and delay
    Lr = L('mmc.mmc_rate.valid_only'); Ld = L('mmc.mmc_delay.filled_or_filtered');
    mmc = struct();
    mmc.params = struct('W', K.mmc_w / fs, 'S', 1 / fs, 'delayW', 4 / fs, 'delayStep', 1 / fs);
    mmc.t = (0:N - 1)' / fs;
    sig = ones(N, 3);
    for j = 1:size(K.mmc_nan, 1), sig(K.mmc_nan(j, 2):K.mmc_nan(j, 3), K.mmc_nan(j, 1)) = NaN; end
    mmc.signal = single(sig);
    ev = false(N, 3);
    for j = 1:size(K.ev_rows, 1), ev(K.ev_rows(j, 2), K.ev_rows(j, 1)) = true; end
    mmc.rate_t = (Lr + (-2:3)') / fs;
    rate = (1:6)' * [1 1 1];
    rate(2, 1) = NaN;
    rate(5, 3) = NaN;
    mmc.firing = struct('events', ev, 'rate', rate, 'peakAmp', rate * 10, 'avgRate', [1 2 3], ...
                        'refractory', 0.05);
    mmc.delay_t = p3(Ld) / fs - (mmc.params.W / 2 - mmc.params.S);
    mmc.delay = [1 1 1; 2 2 2; 3 3 3];
    save(fullfile(d, 'e2_mmc_in_mmc.mat'), 'mmc');
    % one kept slow-wave channel
    Lsw = L('slow_wave.sw_rate.filled_or_filtered');
    slowWaveRateTime = p3(Lsw) / fs; slowWaveRateSeries = [1; 2; 3]; avgSlowWave = 7; %#ok<NASGU>
    rateWinSec = K.sw_w / fs; %#ok<NASGU>
    save(fullfile(d, 'e2_swm_ANT1_slowWaves_ANT1.mat'), 'slowWaveRateTime', ...
         'slowWaveRateSeries', 'avgSlowWave', 'rateWinSec', 'invalidMask');
    % spikes_v2 (v7.3): a struct array (per channel), co-indexed drop, cells, sigma windows
    % and CV2 bins over the channel's own invalid runs
    Lst = L('spikes.spike_times.filled_or_filtered');
    Lw = L('spikes.spike_waveforms.filled_or_filtered');
    Lsg = L('spikes.sigma_windows.valid_only'); Lcv = L('spikes.cv2.valid_only');
    spikes = struct('centers', {p3(Lst) + 1, [Lst + 1; Lst + 2]}, ...
                    'times', {p3(Lst) / fs, [Lst; Lst + 1] / fs}, ...
                    'waveforms', {[1 1; 2 2; 3 3], [4 4; 5 5]}, ...
                    'alignedCenters', {p3(Lw) + 1, [Lw + 1; Lw + 2]}, ...
                    'alignedTimes', {p3(Lw) / fs, [Lw; Lw + 1] / fs}, 'nSpikes', {3, 2}); %#ok<NASGU>
    sigmaWin = struct('centers', {{p3(Lsg) + 1, p3(Lsg) + 1.5}}, ...
                      'sigma', {{[1; 2; 3], [4; 5; 6]}}, 'windowSec', K.sigma_w / fs, ...
                      'stepFrac', 0.5); %#ok<NASGU>
    metrics = struct('cv2_t', {p3(Lcv)' / fs, p3(Lcv)' / fs}, 'cv2_roll', {[1 2 3], [4 5 6]}, ...
                     'nSpikes', {3, 2}); %#ok<NASGU>
    nSamples = K.spk_n; info = struct('P', struct('cv2WinSec', K.cv2_w / fs)); %#ok<NASGU>
    invalidRuns = cell(1, 2);
    for k = 1:2
        invalidRuns{k} = K.spk_invalid(K.spk_invalid(:, 1) == k, 2:3);
    end
    save(fullfile(d, 'e2_spikes_v2.mat'), 'spikes', 'sigmaWin', 'metrics', 'nSamples', ...
         'info', 'invalidRuns', '-v7.3');
    fid = fopen(fullfile(d, 'e2_figure.png'), 'w'); fwrite(fid, 'x'); fclose(fid);
    A = struct('output_rows_before_start', 0, 'start_sample0', i0, 'start_s', i0 / fs);
    cuts = cellfun(@(c) struct('cut', c.cut, 'basis', 'synthetic', ...
        'start_s', (i0 + c.L) / fs, 'start_sample0', i0 + c.L, ...
        'output_rows_before_start', c.L), as_cells(K.cut_L), 'UniformOutput', false);
    rec = struct('epoch_start_sample0', i0, 'electrical_settle_sample0', i0 + 7, ...
                 'analyses', struct('hrv', A, 'breathing', A, 'mmc', A, 'spikes', A, ...
                                    'slow_wave', A), 'cuts', {cuts});
    tr = @(files, cons) night6_recovery_trim_outputs(d, files, cons, rec, OT, fs);
    r = struct();
    r.hrv = tr({'e2_hrv_HRVMeasures.mat', 'e2_hrv_HRBR.mat', 'e2_figure.png', '.', '..'}, {'hrv'});
    r.breathing = tr({'e2_breathing_HRBR.mat'}, {'breathing'});
    r.mmc = tr({'e2_mmc_in_mmc.mat'}, {'mmc'});
    r.spikes = tr({'e2_spikes_v2.mat'}, {'spikes'});
    r.sw = tr({'e2_swm_ANT1_slowWaves_ANT1.mat'}, {'slow_wave'});
    % refusals, by name: trimmed twice; a cut the row lacks; no class; an unknown class; a
    % sibling that already exists; a stamp that does not fit its value
    r.twice = attempt(@() tr({'e2_breathing_HRBR.mat'}, {'breathing'}));
    noCut = rec;
    noCut.cuts = rec.cuts(~cellfun(@(c) strcmp(c.cut, 'hrv.heart_rate.valid_only'), rec.cuts));
    r.no_cut = attempt(@() night6_recovery_trim_outputs(d, {'e2_nocut_HRBR.mat'}, {'hrv'}, ...
                                                        noCut, OT, fs));
    j = find(cellfun(@(x) strcmp(x.file, 'HRBR') && strcmp(x.path, 'heartRateSeries'), OT.vars));
    OT2 = OT;
    OT2.vars{j} = rmfield(OT2.vars{j}, 'trim_class');
    r.no_class = attempt(@() night6_recovery_trim_outputs(d, {'e2_noclass_HRBR.mat'}, ...
                                                          {'hrv'}, rec, OT2, fs));
    OT3 = OT;
    OT3.vars{j}.trim_class = 'probably_valid';
    r.bad_class = attempt(@() night6_recovery_trim_outputs(d, {'e2_noclass_HRBR.mat'}, ...
                                                           {'hrv'}, rec, OT3, fs));
    r.collide = attempt(@() tr({'e2_collide_HRBR.mat'}, {'hrv'}));
    metrics_t = (0:3)' / fs; hrv_series = [1; 2; 3]; %#ok<NASGU>
    save(fullfile(d, 'e2_bad_HRVMeasures.mat'), 'metrics_t', 'hrv_series');
    r.shape = attempt(@() tr({'e2_bad_HRVMeasures.mat'}, {'hrv'}));
    r.marker_variable = OT.markerVariable;
    r.out = struct('hrvm', load(fullfile(d, 'e2_hrv_HRVMeasures.mat')), ...
                   'hrbr', load(fullfile(d, 'e2_hrv_HRBR.mat')), ...
                   'hrbr_b', load(fullfile(d, 'e2_breathing_HRBR.mat')), ...
                   'mmc', load(fullfile(d, 'e2_mmc_in_mmc.mat')), ...
                   'spk', load(fullfile(d, 'e2_spikes_v2.mat')), ...
                   'sw', load(fullfile(d, 'e2_swm_ANT1_slowWaves_ANT1.mat')));
    r.events_class = class(r.out.mmc.mmc.firing.events);
    r.out.mmc.mmc.signal = double(r.out.mmc.mmc.signal);
    r.untouched = struct('nocut', isequal(load(fullfile(d, 'e2_nocut_HRBR.mat')), ...
                                          load(fullfile(d, 'e2_noclass_HRBR.mat'))));
    fid = fopen(fullfile(d, 'e2_spikes_v2.mat'), 'r'); h = fread(fid, [1 128], '*char'); fclose(fid);
    r.spikes_still_v73 = contains(h, 'MATLAB 7.3');
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
