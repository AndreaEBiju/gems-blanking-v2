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
%            reference run on the same masked input, through an oracle (trim_case).
%   unit     night6_recovery_trim_outputs on hand-built files, every convention at L-1/L/L+1.
%   batch    night6_batch's refusals at batch start.
    C = jsondecode(fileread(caseFile));
    out = struct();
    out.plans = cellfun(@plan_case, as_cells(C.plans), 'UniformOutput', false);
    out.readers = cellfun(@reader_case, as_cells(C.readers), 'UniformOutput', false);
    if isfield(C, 'run'), out.run = run_case(C.run); end
    if isfield(C, 'unit'), out.unit = unit_case(C.unit); end
    if isfield(C, 'batch'), out.batch = batch_case(C.batch); end
    if isfield(C, 'trim'), out.trim = trim_case(C.trim); end
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
% Every output file is compared by an oracle built here from the map's definitions.
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
    dD = fullfile(K.out_drop, K.epoch_rel);
    dR = fullfile(K.out_ref, K.epoch_rel);
    files = dir(fullfile(dD, '*.mat'));
    for k = 1:numel(files)
        name = files(k).name;
        T = load(fullfile(dD, name));
        U = load(fullfile(dR, name));
        hit = cellfun(@(f) ~isempty(regexp(name, f.pattern, 'once')), OT.files);
        fr = struct('file', name, 'kind', '', 'rest_equal', false);
        if ~any(hit)
            fr.rest_equal = isequaln(T, U);
            r.files{end + 1} = fr;
            continue
        end
        kind = OT.files{hit}.kind;
        fr.kind = kind;
        cons = run_consumers(name, kind);
        decl = OT.vars(cellfun(@(v) strcmp(v.file, kind) && strcmp(v.role, 'trim'), OT.vars));
        Tr = T; Ur = U;
        for j = 1:numel(decl)
            d = decl{j};
            [res, Tr, Ur] = oracle(T, U, Tr, Ur, d, K, cons, OT);
            res.file = name;
            r.vars{end + 1} = res;
        end
        if strcmp(kind, 'mmc')   % her qc.srcFile names the input file: the out folder
            Tr.mmc.qc.srcFile = ''; Ur.mmc.qc.srcFile = '';
        end
        fr.rest_equal = isequaln(Tr, Ur);   % everything but the trimmed leaves: untouched
        r.files{end + 1} = fr;
    end
end

function c = run_consumers(name, kind)
% The run's consumers from the file name (the wrapper's labels).
    switch kind
        case {'HRBR', 'HRVMeasures'}
            tok = regexp(name, '^e\d+_(.*)_(HRBR|HRVMeasures)\.mat$', 'tokens', 'once');
            c = strsplit(tok{1}, '_');
        case 'spikes_v2', c = {'spikes'};
        case 'slowWaves', c = {'slow_wave'};
        case 'mmc', c = {'mmc'};
    end
end

function [res, Tr, Ur] = oracle(T, U, Tr, Ur, d, K, cons, OT)
    res = struct('path', d.path, 'owner', d.owner, 'action', d.action, 'n_leaves', 0, ...
                 'n_before', 0, 'n_after', 0, 'ok', true, 'why', '');
    vparts = strsplit(d.path, '.');
    sparts = strsplit(d.stamp, '.');
    if ~has_field_path(U, vparts)
        res.why = 'absent';
        return
    end
    if any(strcmp(cons, d.owner))
        L = K.L.(d.owner);
    else
        L = max(cellfun(@(c) K.L.(c), cons));
    end
    leaves = expand(U, vparts);
    sib = numel(vparts) == numel(sparts) && isequal(vparts(1:end - 1), sparts(1:end - 1));
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
                res = check_leaf(res, t{c}, u{c}, st{c}, d, L, K, U, OT);
            end
        else
            res = check_leaf(res, t, u, st, d, L, K, U, OT);
        end
        res.n_leaves = res.n_leaves + 1;
        Tr = subsasgn(Tr, s, []);
        Ur = subsasgn(Ur, s, []);
    end
end

function res = check_leaf(res, t, u, st, d, L, K, U, OT)
    v = double(st(:));
    switch d.convention
        case 'row1', p = v - 1;
        case 'sec0', p = v * K.fs;
        case 'sec_row1', p = v * K.fs - 1;
        case 'sec_xchan_delay'
            W = subsref(U, substruct_path(strsplit(OT.xchanDelayParams{1}, '.')));
            S = subsref(U, substruct_path(strsplit(OT.xchanDelayParams{2}, '.')));
            p = (v + W / 2 - S) * K.fs;
    end
    b = p < L - 1e-6;
    res.n_before = res.n_before + nnz(b);
    res.n_after = res.n_after + nnz(~b);
    if isempty(u) && isempty(t), return, end
    if isvector(u) && numel(u) == numel(b)
        ub = u(~b); tb = []; bb = [];
        switch d.action
            case 'drop', ok = bits_equal(t, ub);
            otherwise
                tb = t(~b); bb = t(b);
                ok = bits_equal(tb, ub);
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
    rec = struct('consumers', struct( ...
        'hrv', struct('output_rows_before_start', Lh), ...
        'breathing', struct('output_rows_before_start', Lb), ...
        'mmc', struct('output_rows_before_start', Lm), ...
        'spikes', struct('output_rows_before_start', Ls)));
    r = struct();
    r.hrv = night6_recovery_trim_outputs(d, {'e2_hrv_HRVMeasures.mat', 'e2_hrv_HRBR.mat', ...
        'e2_figure.png', '.', '..'}, {'hrv'}, rec, RS.outputTimes, fs);
    r.both = night6_recovery_trim_outputs(d, {'e2_hrv_breathing_HRBR.mat'}, ...
        {'hrv', 'breathing'}, rec, RS.outputTimes, fs);
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

function r = batch_case(K)
% night6_batch refuses at batch start, by name, before any recording is loaded.
    r = struct();
    for c = as_cells(K.lists)
        r.(c{1}.name) = attempt(@() night6_batch(c{1}.file, 0, 'DryRun', true));
    end
end
