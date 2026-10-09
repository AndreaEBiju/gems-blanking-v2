function records = night6_run_recording(maskFolder, varargin)
% NIGHT6_RUN_RECORDING  Night 6: run Andrea's analyses on one recording, per-consumer masked.
%
%   records = night6_run_recording(maskFolder, 'GemsRoot', root, 'Units', 'V', ...
%                                  'OutRoot', outRoot)
%
% maskFolder is a store mask folder data/<animal>/<session>/masks/<model-id>/ holding
% one e<start>_masks.mat per epoch (gems_blanking_v2.emit.handoff.write_mask_file).
% Every epoch is analysed ON ITS OWN - never paired with another epoch or recording
% (RULING 2026-10-08 (f) 5: "pre" recordings are baseline-type and never paired).
%
% Per epoch:
%   1. slice the recording by epochStartSample0 / nSamples - the exact sample the mask
%      writer sliced with; seconds are never converted here (night6_prepare_epoch);
%   2. build each consumer's input from the file's columns - raw contacts, the pairs
%      lead "<plus>-<minus>" (mask token <plus>_minus_<minus>), or the software
%      tripole <cuff>_T = 0.5*V1 + 0.5*V3 - V2 - in VOLTS, with that consumer's spans
%      as NaN and nobody else's (night6_consumer_input; invariants 1, 2);
%   3. call, in night6_calls() order:
%        process_dataset_v2         spikes      (her process_dataset steps minus step1b,
%                                                via her bulk_load_one; Andrea 2026-10-09)
%        HR_BR_HRVAnalysis_beats    hrv, breathing (stored beats; batch_process's HR args)
%        slowWaveAnalysis_new       slow_wave   (batch_process / T configuration), one
%                                   ANT channel at a time (Andrea 2026-10-09)
%        extract_mmc                mmc         (ANT1-3 raw, R8; R-peaks = stored beats)
%      Andrea's functions come from processing_new UNCHANGED; skipping, with the reason
%      logged, every consumer in notcomputed_json;
%   4. write night6_record.json beside the outputs: mask folder, model id, mask file
%      hash, code commit, every called function's path and SHA-256, parameters, and
%      per-consumer status. The record is written last; an epoch whose record says
%      "complete" is skipped on a rerun (resumable) unless 'Force' is true.
%
% Name-value inputs (no positional options - invariant 40):
%   GemsRoot    store root; meta.json, the recording (meta.source_path) and the beats
%               file (mask provenance extra.beats_file.hrv_beats) resolve against it
%   Units       'V' | 'mV' | 'uV' of the recording file. REQUIRED, no default (inv. 14)
%   OutRoot     outputs go to OutRoot/<animal>/<session>/<model-id>/<epoch>/
%   MetaFile    default GemsRoot/data/<animal>/<session>/meta.json
%   Figures     Andrea's diagnostic figures (default false)
%   KeepInputs  keep the scratch *_input.mat files the file-based calls read (false)
%   DryRun      plan, build and summarise every input, but call nothing (false)
%   CodeCommit  this repo's commit; default: git rev-parse HEAD beside this file
%   Label       free text copied into each record (e.g. 'STAND-IN smoke run')
%   Force       rerun epochs whose record is complete (false)
    ip = inputParser;
    ip.addRequired('maskFolder', @(x) ischar(x) || isstring(x));
    ip.addParameter('GemsRoot', '', @(x) ischar(x) || isstring(x));
    ip.addParameter('Units', '', @(x) ischar(x) || isstring(x));
    ip.addParameter('OutRoot', '', @(x) ischar(x) || isstring(x));
    ip.addParameter('MetaFile', '', @(x) ischar(x) || isstring(x));
    ip.addParameter('Figures', false, @(x) islogical(x) || isnumeric(x));
    ip.addParameter('KeepInputs', false, @(x) islogical(x) || isnumeric(x));
    ip.addParameter('DryRun', false, @(x) islogical(x) || isnumeric(x));
    ip.addParameter('CodeCommit', '', @(x) ischar(x) || isstring(x));
    ip.addParameter('Label', '', @(x) ischar(x) || isstring(x));
    ip.addParameter('Force', false, @(x) islogical(x) || isnumeric(x));
    ip.parse(maskFolder, varargin{:});
    o = ip.Results;
    for req = {'GemsRoot', 'Units', 'OutRoot'}
        if isempty(o.(req{1}))
            error('night6:required', '%s is required', req{1});
        end
    end
    o = structfun_char(o);
    if isempty(o.CodeCommit), [o.CodeCommit, o.CodeDirty] = code_commit(); else, o.CodeDirty = []; end

    maskFolder = char(o.maskFolder);
    [~, modelId] = fileparts(strip_sep(maskFolder));
    files = dir(fullfile(maskFolder, 'e*_masks.mat'));
    if isempty(files)
        error('night6:noMasks', 'no e<start>_masks.mat in %s', maskFolder);
    end
    starts = cellfun(@(f) sscanf(f, 'e%d_masks.mat'), {files.name});
    [~, order] = sort(starts);
    files = files(order);

    sessionDir = fileparts(fileparts(strip_sep(maskFolder)));
    [animalDir, session] = fileparts(sessionDir);
    [~, animal] = fileparts(animalDir);

    records = cell(1, numel(files));
    todo = true(1, numel(files));
    for k = 1:numel(files)
        tag = erase(files(k).name, '_masks.mat');
        recFile = fullfile(o.OutRoot, animal, session, modelId, tag, 'night6_record.json');
        if ~o.Force && isfile(recFile)
            R = jsondecode(fileread(recFile));
            % Resume only what was made from THIS mask file: a rewritten mask file
            % (same name, new content) must rerun, not be reported as done.
            same = isfield(R, 'mask_file_sha256') && strcmp(R.mask_file_sha256, ...
                sha256_file(fullfile(files(k).folder, files(k).name)));
            if isfield(R, 'status') && strcmp(R.status, 'complete') && same
                records{k} = R;
                todo(k) = false;
                fprintf('[night6] %s %s: complete, skipped (resumable)\n', session, tag);
            elseif isfield(R, 'status') && strcmp(R.status, 'complete')
                fprintf('[night6] %s %s: complete for another mask file, rerun\n', session, tag);
            end
        end
    end
    if ~any(todo), return, end

    metaFile = o.MetaFile;
    if isempty(metaFile)
        metaFile = fullfile(o.GemsRoot, 'data', animal, session, 'meta.json');
    end
    meta = jsondecode(fileread(metaFile));
    if ~strcmp(meta.session, session)
        error('night6:session', 'meta.json is for %s, the mask folder for %s', meta.session, session);
    end
    sigFile = from_root(o.GemsRoot, meta.source_path);
    t0 = tic;
    S = load(sigFile, 'signal', 'fs', 'chanlabels');
    loadS = toc(t0);
    fprintf('[night6] %s: loaded %s (%d x %d) in %.0f s\n', session, sigFile, ...
            size(S.signal, 1), size(S.signal, 2), loadS);

    for k = find(todo)
        tag = erase(files(k).name, '_masks.mat');
        outDir = fullfile(o.OutRoot, animal, session, modelId, tag);
        if isfolder(outDir)
            % Never write beside a previous attempt: stale files would sit next to the
            % new outputs, and an overwritten file would be missed by the outputs
            % diff. The old folder is moved aside (<epoch>.old<k>), not deleted.
            j = 1;
            while isfolder(sprintf('%s.old%d', outDir, j)), j = j + 1; end
            movefile(outDir, sprintf('%s.old%d', outDir, j));
            fprintf('[night6] %s %s: previous attempt moved to %s.old%d\n', session, tag, ...
                    tag, j);
        end
        mkdir(outDir);
        maskFile = fullfile(files(k).folder, files(k).name);
        src = struct('gems_root', o.GemsRoot, 'meta_file', metaFile, 'sig_file', sigFile, ...
                     'sig_load_s', loadS, 'animal', animal, 'session', session, ...
                     'model_id', modelId, 'mask_folder', maskFolder, 'epoch_tag', tag);
        records{k} = run_epoch(maskFile, S, meta, src, outDir, o);
    end
end

% ==========================================================================
function R = run_epoch(maskFile, S, meta, src, outDir, o)
    tEpoch = tic;
    R = struct();
    R.wrapper = 'gems-blanking-v2 matlab/night6 (night6_run_recording)';
    R.status = 'started';
    R.label = o.Label;
    R.code = struct('commit', o.CodeCommit);
    if ~isempty(o.CodeDirty), R.code.dirty = o.CodeDirty; end
    R.source = src;
    R.mask_file = maskFile;
    R.mask_file_sha256 = sha256_file(maskFile);
    R.units = o.Units;
    R.dry_run = logical(o.DryRun);
    R.matlab = version;
    R.started = char(datetime('now', 'TimeZone', 'local', 'Format', 'yyyy-MM-dd''T''HH:mm:ssXXX'));

    M = load(maskFile);
    prov = jsondecode(char(M.provenance_json));
    % Kept verbatim: jsondecode mangles keys such as "ANT1|0-2" into identifiers.
    R.mask_provenance_json = char(M.provenance_json);
    R.mask_gate_json = char(M.gate_json);
    R.stand_in = isfield(prov, 'extra') && isfield(prov.extra, 'STAND_IN');
    if ~strcmp(prov.recording, src.session)
        error('night6:recording', 'mask file is for %s, folder for %s', prov.recording, src.session);
    end
    if isfield(prov, 'extra') && isfield(prov.extra, 'condition')
        R.condition = prov.extra.condition;
    end
    beats = [];
    if isfield(prov, 'extra') && isfield(prov.extra, 'beats_file') ...
            && isfield(prov.extra.beats_file, 'hrv_beats')
        R.source.beats_file = from_root(o.GemsRoot, prov.extra.beats_file.hrv_beats);
        beats = load(R.source.beats_file);
    end

    condition = '';
    if isfield(R, 'condition'), condition = R.condition; end
    plan = night6_prepare_epoch(M, S.chanlabels, meta.channels, size(S.signal, 1), S.fs, ...
                                o.Units, beats, condition);
    R.epoch = struct('start_s', plan.epochStart_s, 'start_sample_0based', plan.i0, ...
                     'n_samples', plan.n, 'fs', plan.fs, ...
                     'rule', ['file samples i0+1..i0+n, i0 = the mask file''s ' ...
                              'epochStartSample0 (never derived from seconds)']);
    R.not_computed = plan.notComputed;
    R.not_measured_mmc = plan.notMeasuredMmc;   % R6: reported, never blanked
    if ~isempty(plan.beats)
        R.beats = struct('n_in_epoch', numel(plan.beats.heartlocs), ...
                         'n_whole_file', plan.beats.nWholeFile, ...
                         'beat_channel', plan.beats.beatChannel, ...
                         'n_blank_spans', size(plan.beats.blankSpans, 1));
    end
    R.consumers = plan.consumers;
    for c = fieldnames(plan.consumers)'
        st = plan.consumers.(c{1});
        if ~strcmp(st.status, 'to_run')
            fprintf('[night6] %s %s: %s %s - %s\n', src.session, src.epoch_tag, c{1}, ...
                    st.status, st.reason);
        end
    end
    R.functions = function_provenance();
    R.params = params();
    % Labels are the epoch tag alone: the folder already names animal, session and
    % model, and Windows MAX_PATH is 260 (cross-platform rule 5) - the smoke run's
    % first version, <session>_<epoch>_spikes_input.mat, reached ~306 and save failed.
    base = src.epoch_tag;
    check_path_budget(outDir, base);
    o.beatsEpochFile = '';
    if ~isempty(plan.beats) && ~isempty(plan.beats.heartlocs) && any(ismember({plan.runs.call}, ...
            {'HR_BR_HRVAnalysis_beats', 'extract_mmc', 'process_dataset_v2'}))
        % These calls read heartlocs + fs from a FILE, 1-based into the signal they are
        % given - so the whole-file beats are re-based to this epoch and written here.
        % (process_dataset_v2 reads them through her bulk_load_one as D.rpeakSamples.)
        o.beatsEpochFile = fullfile(outDir, [base '_beats_epoch.mat']);
        B = plan.beats;
        heartlocs = B.heartlocs; gapAfter = B.gapAfter; blankSpans = B.blankSpans; %#ok<NASGU>
        beatChannel = B.beatChannel; epochStart_s = plan.epochStart_s; fs = plan.fs; %#ok<NASGU>
        epochStartSample0 = plan.i0; sourceBeatsFile = R.source.beats_file; %#ok<NASGU>
        save(o.beatsEpochFile, 'heartlocs', 'fs', 'gapAfter', 'blankSpans', 'beatChannel', ...
             'epochStart_s', 'epochStartSample0', 'sourceBeatsFile');
        R.beats.epoch_file = o.beatsEpochFile;
    end
    R.runs = {};
    failed = false;
    for r = plan.runs
        run = struct('call', r.call, 'consumers', {r.consumers}, 'signals', {r.signals});
        perChannel = ~isempty(r.maskSignal);     % slow_wave, one ANT channel at a time
        if perChannel
            run.mask_signal = r.maskSignal;
            run.keep = r.keep;
        end
        tRun = tic;
        try
            X = night6_consumer_input(plan, S.signal, r);
            run.inputs = summarise(X, r.signals);
            % Invariant 41: an input with no valid sample, or a constant one, is refused
            % by name, never handed to a function that would fail on it (or answer from
            % nothing). Spike channels are independent, so a dead one is dropped and the
            % rest run.
            % A constant input is refused for the spike consumer (process_dataset_v2,
            % Andrea 2026-10-09); the other calls keep the all-NaN refusal they had.
            isSpk = strcmp(r.call, 'process_dataset_v2');
            [empty, why] = dead_columns(X, r.signals, isSpk);
            if isSpk && any(empty) && ~all(empty)
                run.dropped_no_valid_samples = r.signals(empty);
                R.consumers.spikes.dropped_no_valid_samples = r.signals(empty);
                R.consumers.spikes.reason = sprintf('dropped: %s', why);
                X = X(:, ~empty);
                r.signals = r.signals(~empty);
            end
            [empty, why] = dead_columns(X, r.signals, isSpk);
            if any(empty)
                run.status = 'skipped_no_valid_samples';
                run.reason = why;
                R = set_status(R, r, perChannel, 'skipped_no_valid_samples', why);
                fprintf('[night6] %s %s: %s [%s] skipped - %s\n', src.session, ...
                        src.epoch_tag, r.call, strjoin(r.consumers, ','), why);
            else
                if ~o.DryRun
                    d0 = dir(outDir);
                    [run.condition_label, extra] = call_one(r, X, plan, base, outDir, o);
                    d1 = dir(outDir);
                    run.outputs = setdiff({d1.name}, {d0.name});
                    for fn = fieldnames(extra)', run.(fn{1}) = extra.(fn{1}); end
                end
                run.status = 'ok';
                R = set_status(R, r, perChannel, ternary(o.DryRun, 'planned', 'ran'), '');
            end
        catch ME
            failed = true;
            run.status = 'failed';
            run.error = sprintf('%s: %s', ME.identifier, ME.message);
            R = set_status(R, r, perChannel, 'failed', run.error);
            fprintf(2, '[night6] %s %s: %s FAILED - %s\n', src.session, src.epoch_tag, ...
                    r.call, run.error);
        end
        run.wall_s = toc(tRun);
        fprintf('[night6] %s %s: %s [%s] %s in %.0f s\n', src.session, src.epoch_tag, ...
                r.call, strjoin(r.consumers, ','), run.status, run.wall_s);
        R.runs{end + 1} = run;
        clear X
        close all force
    end
    R = slow_wave_status(R);
    R.wall_s = toc(tEpoch);
    R.finished = char(datetime('now', 'TimeZone', 'local', 'Format', 'yyyy-MM-dd''T''HH:mm:ssXXX'));
    R.status = ternary(failed, 'failed', ternary(o.DryRun, 'dry_run', 'complete'));
    write_json(fullfile(outDir, 'night6_record.json'), R);
end

function [label, extra] = call_one(r, X, plan, base, outDir, o)
% The one place the consumers' calls are made. Arguments by position follow Andrea's own
% drivers exactly (invariant 40); see params() for where each value comes from.
% extra: anything the run record should carry from the call.
    P = params();
    figs = logical(o.Figures);
    fs = plan.fs;
    extra = struct();
    switch r.call
        case 'process_dataset_v2'
            label = base;
            f = write_input(fullfile(outDir, [label '_spikes_in.mat']), X, fs);
            cleanup = onCleanup(@() drop(f, o.KeepInputs)); %#ok<NASGU>
            % Her headless loader builds D exactly as her bulk driver does: yOut in
            % volts with the masked samples NaN, R-peaks from the epoch's beats file
            % (heartlocs, 1-based samples into this input) or none.
            w = warning('off', 'MATLAB:load:variableNotFound');   % no t in the input
            restoreW = onCleanup(@() warning(w)); %#ok<NASGU>
            D = bulk_load_one(f, o.beatsEpochFile, struct('neuralCols', 1:numel(r.signals), ...
                                                         'labels', {r.signals}));
            D.condition = plan.condition;
            [D, info] = process_dataset_v2(D, 'PlotMode', figs, 'NanPadMs', P.spikes.nanPadMs);
            save_spikes_v2(fullfile(outDir, [label '_spikes_v2.mat']), D, info, r.signals);
            extra.n_rpeaks = info.n_rpeaks;
            extra.spike_check = info.spike_check;
            if figs
                save_all_figures(outDir, [label '_spikes_v2'], {'png', 'fig'});   % hers
            end
        case 'HR_BR_HRVAnalysis_beats'
            label = [base '_' strjoin(r.consumers, '_')];
            B = plan.beats;
            H = P.hr;
            HR_BR_HRVAnalysis_beats(o.beatsEpochFile, X, fs, H.cutoff, H.order, outDir, label, 1, ...
                [], H.edgeBufferSec, H.winSec, H.stepSec, figs, H.hrBrWinSec, ...
                'GapAfter', B.gapAfter, 'BlankSpans', B.blankSpans);
        case 'slowWaveAnalysis_new'
            % One ANT channel at a time: X carries mask r.maskSignal on all three
            % columns; only the kept channels' outputs survive (night6_keep_slow_wave).
            label = sprintf('%s_swm_%s', base, r.maskSignal);
            W = P.slow_wave;
            d0 = dir(outDir);
            slowWaveAnalysis_new(X, W.lowPassOn, W.lowPassCutoff, W.lowPassOrder, fs, ...
                W.smoothWindow, figs, outDir, label, [], W.edgeBufferSec);
            d1 = dir(outDir);
            extra.slow_wave = night6_keep_slow_wave(outDir, label, base, r.signals, r.keep, ...
                r.maskSignal, setdiff({d1.name}, {d0.name}));
        case 'extract_mmc'
            label = base;
            f = write_input(fullfile(outDir, [label '_mmc_in.mat']), X, fs);
            cleanup = onCleanup(@() drop(f, o.KeepInputs)); %#ok<NASGU>
            % R-peaks by file + DECLARED unit, checked (RULING 2026-10-08 (g) 2).
            extract_mmc(f, o.beatsEpochFile, night6_mmc_opts(o.beatsEpochFile, fs, plan.n));
        otherwise
            error('night6:call', 'no call %s', r.call);
    end
end

function P = params()
% Every argument handed to the calls, and where it came from.
    P.spikes = struct('method', 'process_dataset_v2', 'steps', {night6_v2_steps()}, ...
                      'bandpassLow', 300, 'bandpassHigh', 3000, 'nanPadMs', 5, ...
                      'source', ['Andrea 2026-10-09: P = pipeline_params(); P.bandpassLow = ' ...
                                 '300; P.bandpassHigh = 3000; everything else default ' ...
                                 '(threshSigma 4.5); her process_dataset steps minus ' ...
                                 'step1b; D from her bulk_load_one with D.rpeakSamples']);
    P.hr = struct('cutoff', 8, 'order', 4, 'edgeBufferSec', 0.75, 'winSec', 20, ...
                  'stepSec', 1, 'hrBrWinSec', 60, 'chanidx', 1, ...
                  'source', 'batch_process.m P.hr_* (= run_continuous.m, = T)');
    P.slow_wave = struct('lowPassOn', true, 'lowPassCutoff', 0.15, 'lowPassOrder', 2, ...
                         'smoothWindow', 5, 'edgeBufferSec', 15, ...
                         'source', ['batch_process.m P.sw_* (= T / tolerance_sweep); NOT ' ...
                                    'run_continuous.m (lowPassOn false, 2 Hz, order 4, window 10, buffer 3)']);
    P.mmc = struct('gastricCols', [1 2 3], 'rpeak_source', ...
                   ['epoch beats file: rpeakVar heartlocs, rpeakUnits samples, rpeakFs fs ' ...
                    '(night6_mmc_opts, unit checked)'], 'other', 'extract_mmc defaults');
end

function F = function_provenance()
% Andrea's functions must resolve to processing_new - a stray copy would shadow them -
% and the wrapper's own (process_dataset_v2 and its step list) to this folder.
    F = struct();
    C = night6_calls();
    ours = {'process_dataset_v2', 'night6_v2_steps'};
    hers = [setdiff({C.name}, ours, 'stable'), night6_v2_steps(), ...
            {'pipeline_params', 'bulk_load_one'}];
    here = fileparts(mfilename('fullpath'));
    for name = [hers, ours]
        p = which(name{1});
        [d, ~] = fileparts(p);
        [~, leaf] = fileparts(d);
        if ismember(name{1}, ours)
            if ~strcmp(d, here)
                error('night6:shadow', '%s resolves to %s, not this wrapper (%s)', name{1}, p, here);
            end
        elseif ~strcmp(leaf, 'processing_new')
            error('night6:shadow', '%s resolves to %s, not processing_new', name{1}, p);
        end
        F.(name{1}) = struct('path', p, 'sha256', sha256_file(p));
    end
    % Not called, but params() copies its constants (HR, slow wave): its hash says
    % which version of the driver the parameters were taken from.
    p = which('batch_process');
    F.batch_process = struct('path', p, 'sha256', sha256_file(p), 'role', 'parameter source');
end

function [dead, why] = dead_columns(X, signals, checkConstant)
% Columns with no finite sample - or, with checkConstant, whose finite samples are all
% equal (invariant 41) - and one sentence naming them.
    noValid = all(isnan(X), 1);
    constant = false(size(noValid));
    if checkConstant
        hi = max(X, [], 1, 'omitnan');
        lo = min(X, [], 1, 'omitnan');
        constant = ~noValid & hi == lo;
    end
    dead = noValid | constant;
    parts = {};
    if any(noValid)
        parts{end + 1} = sprintf('no valid sample in [%s]: blanked for the whole epoch', ...
                                 strjoin(signals(noValid), ' '));
    end
    if any(constant)
        parts{end + 1} = sprintf('constant input in [%s]: no signal to analyse', ...
                                 strjoin(signals(constant), ' '));
    end
    why = strjoin(parts, '; ');
end

function R = set_status(R, r, perChannel, status, reason)
% A run's outcome for its consumers. slow_wave runs one ANT channel at a time, so its
% outcome is recorded per kept channel and summarised once after the loop.
    if perChannel
        for s = r.keep
            st = struct('status', status, 'mask_signal', r.maskSignal);
            if ~isempty(reason), st.reason = reason; end
            R.consumers.slow_wave.channels.(matlab.lang.makeValidName(s{1})) = st;
        end
        return
    end
    for c = r.consumers
        R.consumers.(c{1}).status = status;
        if ~isempty(reason), R.consumers.(c{1}).reason = reason; end
    end
end

function R = slow_wave_status(R)
% slow_wave's overall status from its channels: failed if any failed, else ran/planned
% if any did, else skipped (every channel's reason is in .channels).
    if ~isfield(R.consumers, 'slow_wave') || ~isfield(R.consumers.slow_wave, 'channels')
        return
    end
    ch = struct2cell(R.consumers.slow_wave.channels);
    st = cellfun(@(c) c.status, ch, 'UniformOutput', false);
    order = {'failed', 'ran', 'planned', 'skipped_no_valid_samples'};
    for k = 1:numel(order)
        if any(strcmp(st, order{k}))
            R.consumers.slow_wave.status = order{k};
            break
        end
    end
    R.consumers.slow_wave.reason = 'per ANT channel; see channels (Andrea, 2026-10-09)';
end

function s = summarise(X, signals)
% What each input looked like: NaN accounting and the first valid sample, so the
% record shows exactly what the consumer was handed. Absent keys, never null.
    s = cell(1, size(X, 2));
    for j = 1:size(X, 2)
        b = isnan(X(:, j));
        e = struct('signal', signals{j}, 'n_nan', nnz(b), ...
                   'n_nan_runs', nnz(diff([false; b]) == 1));
        fv = find(~b, 1);
        if ~isempty(fv)
            e.first_valid_row = fv;
            e.first_valid_value_V = X(fv, j);
        end
        if any(b)
            e.first_nan_row = find(b, 1, 'first');
            e.last_nan_row = find(b, 1, 'last');
            d = diff([false; b; false]);
            e.nan_runs = [find(d == 1), find(d == -1) - 1];   % 1-based inclusive rows
        end
        s{j} = e;
    end
end

function check_path_budget(outDir, base)
% Fail with a clear message, before any work, if the deepest file this epoch can
% write would exceed Windows MAX_PATH. PATH_MARGIN covers the longest suffix the
% calls append to a label (measured in the smoke run; see the commit message).
    PATH_MARGIN = 60;
    deepest = numel(fullfile(outDir, base)) + PATH_MARGIN;
    if ispc && deepest > 259
        error('night6:maxPath', ['outputs under %s could reach %d characters, over ' ...
              'Windows MAX_PATH (260): choose a shorter OutRoot'], outDir, deepest);
    end
end

function save_spikes_v2(f, D, info, signals)
% process_dataset_v2's outputs: everything her steps added to D except the per-sample
% arrays (D.y, D.t, D.filtered, D.sigma, D.cardiacBlank - each as long as the epoch),
% plus each channel's invalid samples as 1-based inclusive runs (validMask inverted).
    keep = intersect(fieldnames(D), {'fs', 'neuralChannels', 'channelLabels', ...
        'rpeakSamples', 'rpeakTimes', 'condition', 'cardiacBlankWinMs', 'bandInfo', ...
        'noiseInfo', 'sigmaWin', 'spikes', 'detectInfo', 'envelope', 'modality', 'metrics'});
    out = struct();
    for k = keep(:)', out.(k{1}) = D.(k{1}); end
    out.invalidRuns = cell(1, size(D.validMask, 2));
    for k = 1:size(D.validMask, 2)
        d = diff([false; ~D.validMask(:, k); false]);
        out.invalidRuns{k} = [find(d == 1), find(d == -1) - 1];
    end
    out.signals = signals;
    out.info = info;
    out.nSamples = size(D.validMask, 1);
    save(f, '-struct', 'out', '-v7.3');
end

function f = write_input(f, X, fs)
% A blankmotion-shaped scratch file for the two file-based calls (as tolerance_sweep
% writes): yOut in volts with the masked samples NaN, and fs. No t: extract_mmc picks
% the largest matrix, which must be yOut.
    yOut = X; removedSegmentIdx = zeros(0, 2); removedSegments = zeros(0, 2); %#ok<NASGU>
    blankingApplied = true; hrChanIdx = uint8(1); %#ok<NASGU>
    save(f, 'yOut', 'fs', 'removedSegmentIdx', 'removedSegments', 'blankingApplied', ...
         'hrChanIdx', '-v7.3', '-nocompression');
end

function drop(f, keep)
    if ~keep && isfile(f), delete(f); end
end

function p = from_root(root, rel)
% Store paths are relative POSIX (cross-platform rule 2); resolve them here.
    parts = strsplit(char(rel), '/');
    p = fullfile(char(root), parts{:});
end

function s = strip_sep(p)
    s = char(p);
    while ~isempty(s) && any(s(end) == '/\'), s = s(1:end - 1); end
end

function o = structfun_char(o)
    for f = fieldnames(o)'
        if isstring(o.(f{1})), o.(f{1}) = char(o.(f{1})); end
    end
end

function [c, dirty] = code_commit()
    here = fileparts(mfilename('fullpath'));
    [st, out] = system(sprintf('git -C "%s" rev-parse HEAD', here));
    if st ~= 0
        error('night6:commit', 'cannot read the code commit (pass CodeCommit): %s', out);
    end
    c = strtrim(out);
    [~, out] = system(sprintf('git -C "%s" status --porcelain --untracked-files=no', here));
    dirty = ~isempty(strtrim(out));
end

function h = sha256_file(f)
    fid = fopen(f, 'r');
    b = fread(fid, inf, '*uint8');
    fclose(fid);
    md = java.security.MessageDigest.getInstance('SHA-256');
    md.update(typecast(b, 'int8'));
    h = lower(reshape(dec2hex(typecast(md.digest(), 'uint8'), 2)', 1, []));
end

function write_json(f, R)
% UTF-8, \n line endings, atomic (cross-platform rules 10, 11). jsonencode writes a
% NaN or Inf as null; a missing value must be an absent key, so any non-finite
% number anywhere in the record is refused before encoding.
    assert_finite(R, 'record', f);
    txt = jsonencode(R, 'PrettyPrint', true);
    tmp = [f '.tmp'];
    fid = fopen(tmp, 'w', 'n', 'UTF-8');
    fwrite(fid, strrep(txt, sprintf('\r\n'), newline), 'char');
    fclose(fid);
    movefile(tmp, f, 'f');
end

function assert_finite(v, where, f)
    if isnumeric(v) && ~all(isfinite(v(:)))
        error('night6:nonFinite', 'refusing to write NaN/Inf at %s into %s', where, f);
    elseif isstruct(v)
        for i = 1:numel(v)
            for fn = fieldnames(v)'
                assert_finite(v(i).(fn{1}), sprintf('%s(%d).%s', where, i, fn{1}), f);
            end
        end
    elseif iscell(v)
        for i = 1:numel(v)
            assert_finite(v{i}, sprintf('%s{%d}', where, i), f);
        end
    end
end

function o = ternary(c, a, b)
    if c, o = a; else, o = b; end
end
