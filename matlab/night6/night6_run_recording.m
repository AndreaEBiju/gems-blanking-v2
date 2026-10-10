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
%   BeatsRoot   optional, PILOT RUNS ONLY (PilotRoot; else 'night6:beatsRoot', review
%               2026-10-10 fix 3): the root the beats file resolves against INSTEAD of
%               GemsRoot (the same relative path data/<animal>/<session>/<session>_beats.mat),
%               for a routed train not yet published beside meta.json - e.g. the pilot's
%               byte-identical copies from the routing workspace. The file must still
%               hash to the mask provenance's sha256 (night6:beatsSha), and it is hashed
%               before AND after it is loaded, so the bytes loaded are the bytes hashed
%               (a change in between: 'night6:beatsChanged'); the root used is in every
%               record (source.beats_root). Default '' (GemsRoot).
%   PilotRoot   the declared pilot folder of a PILOT run (night6_pilot_root: it must exist,
%               must not be or sit under a junction or symbolic link, and must hold
%               OutRoot - else 'night6:pilotRoot'). Default '' (not a pilot run). Only a
%               pilot run reads a mask made on the STAND-IN tolerance table (provenance
%               extra.TOLERANCE_STAND_IN, RULING 2026-10-09 (g) 7); outside one such a
%               mask folder is refused by name before the recording is loaded
%               ('night6:toleranceStandIn', review 2026-10-10 fix 1). Every record carries
%               stand_in (extra.STAND_IN or extra.TOLERANCE_STAND_IN), tolerance_stand_in
%               (the label, when there is one) and pilot_root (in a pilot run).
%   Units       'V' | 'mV' | 'uV' of the recording file. REQUIRED, no default (inv. 14)
%   OutRoot     outputs go to OutRoot/<animal>/<session>/<model-id>/<epoch>/
%   MetaFile    default GemsRoot/data/<animal>/<session>/meta.json
%   Figures     Andrea's diagnostic figures (default false)
%   KeepInputs  keep the scratch *_input.mat files the file-based calls read (false)
%   DryRun      plan, build and summarise every input, but call nothing (false)
%   CodeCommit  this repo's commit; default: git rev-parse HEAD beside this file
%   Label       free text copied into each record (e.g. 'STAND-IN smoke run')
%   Force       rerun epochs whose record is complete (false)
%   Step1aFallback  the declared animal x cuff list whose spike consumer falls back to
%               step1a (RULING 2026-10-08 (j) 1; night6_step1a_fallback). Default:
%               step1a_fallback.json beside this file, which is empty. Its path, hash and
%               keys go into every record; an unreadable list stops the run.
%   RecoveryStarts  the declared recovery-starts file (RULING 2026-10-08 (k) 2; written by
%               gems_blanking_v2.extent.recovery_start.write_recovery_starts). Every
%               stim_recovery epoch REQUIRES it: every analysis's input is masked
%               before the file's electrical settling (night6_recovery_lead_in), each
%               output variable is cut at its own cut, and an analysis with no start is
%               refused by name. Its path and SHA-256 go into the record; a changed file reruns the
%               stim_recovery epochs on resume. Default '' (no file: stim_recovery refused).
%   RecoveryTrimMode  REQUIRED, no default (night6_trim_modes): mode (B),
%               'mask_to_electrical_drop_outputs' (RULING 2026-10-09 item 6), masks every
%               input up to the electrical settling and then cuts every output VARIABLE
%               at its own cut by its class, adds each windowed value's valid fraction
%               and recomputes the whole-epoch averages (night6_recovery_trim_outputs, by
%               the starts file's output time map). A missing, unknown or withdrawn mode
%               ((A) 'mask_to_own_start') is refused by name before any work; the mode is
%               in every record, and a stim_recovery epoch made under another mode reruns.
%   SlowWaveRate  REQUIRED, no default (night6_slow_wave_rates; RULING 2026-10-09 (c) 6):
%               'full' or 'decimated78', the rate slowWaveAnalysis_new runs at. Either way
%               each run passes its masked spans as blankIdx as well as NaN
%               (night6_call_slow_wave). Missing or unknown is refused by name before any
%               work; the rate is in every record, and an epoch made at another rate reruns.
    ip = inputParser;
    ip.addRequired('maskFolder', @(x) ischar(x) || isstring(x));
    ip.addParameter('GemsRoot', '', @(x) ischar(x) || isstring(x));
    ip.addParameter('BeatsRoot', '', @(x) ischar(x) || isstring(x));
    ip.addParameter('PilotRoot', '', @(x) ischar(x) || isstring(x));
    ip.addParameter('Units', '', @(x) ischar(x) || isstring(x));
    ip.addParameter('OutRoot', '', @(x) ischar(x) || isstring(x));
    ip.addParameter('MetaFile', '', @(x) ischar(x) || isstring(x));
    ip.addParameter('Figures', false, @(x) islogical(x) || isnumeric(x));
    ip.addParameter('KeepInputs', false, @(x) islogical(x) || isnumeric(x));
    ip.addParameter('DryRun', false, @(x) islogical(x) || isnumeric(x));
    ip.addParameter('CodeCommit', '', @(x) ischar(x) || isstring(x));
    ip.addParameter('Label', '', @(x) ischar(x) || isstring(x));
    ip.addParameter('Force', false, @(x) islogical(x) || isnumeric(x));
    ip.addParameter('Step1aFallback', fullfile(fileparts(mfilename('fullpath')), ...
                    'step1a_fallback.json'), @(x) ischar(x) || isstring(x));
    ip.addParameter('RecoveryStarts', '', @(x) ischar(x) || isstring(x));
    ip.addParameter('RecoveryTrimMode', '', @(x) ischar(x) || isstring(x));
    ip.addParameter('SlowWaveRate', '', @(x) ischar(x) || isstring(x));
    ip.parse(maskFolder, varargin{:});
    o = ip.Results;
    for req = {'GemsRoot', 'Units', 'OutRoot'}
        if isempty(o.(req{1}))
            error('night6:required', '%s is required', req{1});
        end
    end
    o = structfun_char(o);
    o.PilotRoot = night6_pilot_root(o.PilotRoot, o.OutRoot);   % '' unless a pilot run
    if ~isempty(o.BeatsRoot) && isempty(o.PilotRoot)
        error('night6:beatsRoot', ['BeatsRoot %s is for pilot runs only (review 2026-10-10 ' ...
              'fix 3): no PilotRoot is declared'], o.BeatsRoot);
    end
    if isempty(o.CodeCommit), [o.CodeCommit, o.CodeDirty] = code_commit(); else, o.CodeDirty = []; end
    o.fallback = night6_step1a_fallback(o.Step1aFallback);   % read before any work
    o.RecoveryTrimMode = night6_check_trim_mode(o.RecoveryTrimMode);   % (k) 2: required
    o.RS = night6_recovery_start(o.RecoveryStarts);          % (k) 2; [] when not declared
    [~, o.SW] = night6_slow_wave_rates(o.SlowWaveRate);       % (c) 6: required, by name

    maskFolder = char(o.maskFolder);
    [~, modelId] = fileparts(strip_sep(maskFolder));
    files = dir(fullfile(maskFolder, 'e*_masks.mat'));
    if isempty(files)
        error('night6:noMasks', 'no e<start>_masks.mat in %s', maskFolder);
    end
    starts = cellfun(@(f) sscanf(f, 'e%d_masks.mat'), {files.name});
    [~, order] = sort(starts);
    files = files(order);
    % Review 2026-10-10 fix 1: a mask made on the STAND-IN tolerance table is read only in a
    % declared pilot run - refused by name before anything is loaded or resumed.
    if isempty(o.PilotRoot)
        for k = 1:numel(files)
            Mp = load(fullfile(files(k).folder, files(k).name), 'provenance_json');
            if ~isfield(Mp, 'provenance_json'), continue, end
            tol = tolerance_stand_in(jsondecode(char(Mp.provenance_json)));
            if ~isempty(tol)
                error('night6:toleranceStandIn', ['%s was made on the stand-in tolerance ' ...
                      'table (%s): read only in a declared pilot run (PilotRoot; RULING ' ...
                      '2026-10-09 (g) 7, "written to the pilot folder only")'], ...
                      fullfile(files(k).folder, files(k).name), tol);
            end
        end
    end

    sessionDir = fileparts(fileparts(strip_sep(maskFolder)));
    [animalDir, session] = fileparts(sessionDir);
    [~, animal] = fileparts(animalDir);

    records = cell(1, numel(files));
    todo = true(1, numel(files));
    EdNow = night6_edge_settling();   % the resume key's edge-settling hash (review 2026-10-09)
    for k = 1:numel(files)
        tag = erase(files(k).name, '_masks.mat');
        recFile = fullfile(o.OutRoot, animal, session, modelId, tag, 'night6_record.json');
        if ~o.Force && isfile(recFile)
            R = jsondecode(fileread(recFile));
            % Resume only what was made from THIS mask file: a rewritten mask file
            % (same name, new content) must rerun, not be reported as done.
            same = isfield(R, 'mask_file_sha256') && strcmp(R.mask_file_sha256, ...
                night6_sha256_file(fullfile(files(k).folder, files(k).name))) ...
                && same_recovery_start(R, o.RS, o.RecoveryTrimMode) ...
                && isfield(R, 'slow_wave_rate') && isstruct(R.slow_wave_rate) ...
                && strcmp(R.slow_wave_rate.name, o.SW.name) ...   % (c) 6: same rate
                && isfield(R, 'edge_settling') && isstruct(R.edge_settling) ...
                && isfield(R.edge_settling, 'sha256') ...
                && strcmp(R.edge_settling.sha256, EdNow.sha256);   % same edge settlings
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
    R.mask_file_sha256 = night6_sha256_file(maskFile);
    R.units = o.Units;
    R.recovery_trim_mode = o.RecoveryTrimMode;   % (k) 2: declared, never defaulted
    R.slow_wave_rate = struct('name', o.SW.name, 'factors', o.SW.factors, ...
                              'factor', o.SW.factor);   % (c) 6: declared, never defaulted
    R.dry_run = logical(o.DryRun);
    R.matlab = version;
    R.started = char(datetime('now', 'TimeZone', 'local', 'Format', 'yyyy-MM-dd''T''HH:mm:ssXXX'));

    M = load(maskFile);
    prov = jsondecode(char(M.provenance_json));
    % Kept verbatim: jsondecode mangles keys such as "ANT1|0-2" into identifiers.
    R.mask_provenance_json = char(M.provenance_json);
    R.mask_gate_json = char(M.gate_json);
    tol = tolerance_stand_in(prov);   % review 2026-10-10 fix 1: either mark is a stand-in
    R.stand_in = (isfield(prov, 'extra') && isfield(prov.extra, 'STAND_IN')) || ~isempty(tol);
    if ~isempty(tol)
        if isempty(o.PilotRoot)   % refused before loading; kept here for a direct caller
            error('night6:toleranceStandIn', '%s: stand-in tolerance mask outside a pilot run', ...
                  maskFile);
        end
        R.tolerance_stand_in = tol;
    end
    if ~isempty(o.PilotRoot), R.pilot_root = o.PilotRoot; end
    if ~strcmp(prov.recording, src.session)
        error('night6:recording', 'mask file is for %s, folder for %s', prov.recording, src.session);
    end
    if isfield(prov, 'extra') && isfield(prov.extra, 'condition')
        R.condition = prov.extra.condition;
    end
    beats = [];
    if isfield(prov, 'extra') && isfield(prov.extra, 'beats_file') ...
            && isfield(prov.extra.beats_file, 'hrv_beats')
        % The record (gems_blanking_v2.emit.hr_beats.beats_file_record, written by the one
        % train resolver) names the train's store path, its sha256, its origin as a 0-based
        % file sample and the epoch's beat count; night6_prepare_epoch slices with them.
        ref = prov.extra.beats_file;
        broot = o.GemsRoot;
        if ~isempty(o.BeatsRoot), broot = o.BeatsRoot; end
        R.source.beats_root = broot;
        R.source.beats_file = from_root(broot, ref.hrv_beats);
        if ~isfile(R.source.beats_file)
            error('night6:beatsMissing', ['the mask provenance names the beat train %s, which ' ...
                  'is not under %s (not yet published?)'], ref.hrv_beats, broot);
        end
        % Review 2026-10-10 fix 3: the bytes loaded are the bytes hashed - hashed before and
        % after the load, refused if they differ (the sha256 is then checked against the
        % mask provenance's in night6_prepare_epoch, night6:beatsSha).
        h0 = night6_sha256_file(R.source.beats_file);
        data = load(R.source.beats_file);
        h1 = night6_sha256_file(R.source.beats_file);
        if ~strcmp(h0, h1)
            error('night6:beatsChanged', ['the beats file %s changed while it was loaded ' ...
                  '(sha256 %s before, %s after)'], R.source.beats_file, h0, h1);
        end
        beats = struct('data', data, 'record', ref, 'sha256', h0);
    end

    condition = '';
    if isfield(R, 'condition'), condition = R.condition; end
    plan = night6_prepare_epoch(M, S.chanlabels, meta.channels, size(S.signal, 1), S.fs, ...
                                o.Units, beats, condition, ...
                                'Recovery', struct('starts', o.RS, 'session', src.session, ...
                                                   'mode', o.RecoveryTrimMode));
    R.recovery_start = plan.recoveryStart;   % (k) 2: each analysis's start, basis, source
    R.epoch = struct('start_s', plan.epochStart_s, 'start_sample_0based', plan.i0, ...
                     'n_samples', plan.n, 'fs', plan.fs, ...
                     'rule', ['file samples i0+1..i0+n, i0 = the mask file''s ' ...
                              'epochStartSample0 (never derived from seconds)']);
    R.not_computed = plan.notComputed;
    if ~isempty(plan.periR)   % review fix 2: the spike mask's peri-R state, train or none
        R.peri_r = plan.periR;
    end
    R.not_measured_mmc = plan.notMeasuredMmc;   % R6: reported, never blanked
    if ~isempty(plan.noHeartRef)   % review 2026-10-10 fix 4: (g) 1's flags, checked present
        R.no_heartbeat_reference = plan.noHeartRef;
    end
    if ~isempty(plan.beats)
        R.beats = struct('n_in_epoch', numel(plan.beats.heartlocs), ...
                         'n_whole_file', plan.beats.nWholeFile, ...
                         'beat_channel', plan.beats.beatChannel, ...
                         'n_blank_spans', size(plan.beats.blankSpans, 1), ...
                         'origin_sample0', plan.beats.originSample0, ...
                         'sha256', plan.beats.sha256);
    end
    R.consumers = plan.consumers;
    for c = fieldnames(plan.consumers)'
        st = plan.consumers.(c{1});
        if ~strcmp(st.status, 'to_run')
            fprintf('[night6] %s %s: %s %s - %s\n', src.session, src.epoch_tag, c{1}, ...
                    st.status, st.reason);
        end
    end
    R.step1a_fallback = struct('file', o.fallback.file, 'sha256', o.fallback.sha256, ...
                               'keys', {o.fallback.keys});
    o.animal = src.animal;
    R.functions = function_provenance();
    R.params = params();
    % RULING 2026-10-09 (c) 1-2: the measured edge settlings this build applies (the spike
    % edge pad here; the mmc 1.5 s pad in the masks' extents), with the measurement files'
    % SHA-256 - the whole declaration, so the record names what every pad rests on.
    Ed = night6_edge_settling();
    R.edge_settling = struct('declaration', Ed.file, 'sha256', Ed.sha256, ...
                             'spikes', Ed.spikes, 'mmc', Ed.mmc);
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
        sourceOriginSample0 = B.originSample0; sourceSha256 = B.sha256; %#ok<NASGU>
        save(o.beatsEpochFile, 'heartlocs', 'fs', 'gapAfter', 'blankSpans', 'beatChannel', ...
             'epochStart_s', 'epochStartSample0', 'sourceBeatsFile', 'sourceOriginSample0', ...
             'sourceSha256');
        R.beats.epoch_file = o.beatsEpochFile;
    end
    R.runs = {};
    failed = false;
    for r = plan.runs
        run = struct('call', r.call, 'consumers', {r.consumers}, 'signals', {r.signals});
        if strcmp(r.call, 'HR_BR_HRVAnalysis_beats')   % (i) 4: what this call is read for
            run.outputs_used = night6_hr_outputs(r.consumers);
        end
        perChannel = ~isempty(r.maskSignal);     % slow_wave, one ANT channel at a time
        if perChannel
            run.mask_signal = r.maskSignal;
            run.keep = r.keep;
        end
        tRun = tic;
        try
            X = night6_consumer_input(plan, S.signal, r);
            run.inputs = summarise(X, r.signals);
            % Invariant 41, for EVERY call: an input column with no valid sample, or a
            % constant one (an exactly-zero column included), is refused by name, never
            % handed to a function that would fail on it or answer from nothing.
            %   spikes     channels are independent: a dead one is dropped, the rest run;
            %   slow_wave  a dead KEPT channel is refused by name and the run keeps the
            %              others; a dead column that is not kept is named
            %              (dead_not_kept) - night6_keep_slow_wave discards its output;
            %   HR, mmc    a dead column refuses the whole call.
            isSpk = strcmp(r.call, 'process_dataset_v2');
            [empty, why] = dead_columns(X, r.signals);
            if isSpk && any(empty) && ~all(empty)
                run.dropped_no_valid_samples = r.signals(empty);
                R.consumers.spikes.dropped_no_valid_samples = r.signals(empty);
                R.consumers.spikes.reason = sprintf('dropped: %s', why);
                X = X(:, ~empty);
                r.signals = r.signals(~empty);
                [empty, why] = dead_columns(X, r.signals);
            elseif perChannel && any(empty)
                isKept = ismember(r.signals, r.keep);
                deadKept = r.signals(empty & isKept);
                if ~isempty(deadKept) && numel(deadKept) < numel(r.keep)
                    [~, whyKept] = dead_columns(X(:, empty & isKept), deadKept);
                    refused = r;
                    refused.keep = deadKept;
                    R = set_status(R, refused, true, 'skipped_no_valid_samples', whyKept);
                    run.refused_keep = deadKept;
                    run.refused_reason = whyKept;
                    r.keep = setdiff(r.keep, deadKept, 'stable');
                    run.keep = r.keep;
                    fprintf('[night6] %s %s: %s [%s] refused - %s\n', src.session, ...
                            src.epoch_tag, r.call, strjoin(deadKept, ','), whyKept);
                end
                if numel(deadKept) < nnz(isKept)
                    if any(empty & ~isKept)
                        [~, run.dead_not_kept_reason] = dead_columns(X(:, empty & ~isKept), ...
                                                                    r.signals(empty & ~isKept));
                        run.dead_not_kept = r.signals(empty & ~isKept);
                    end
                    empty = false(size(empty));   % the kept channels that remain run
                else
                    [~, why] = dead_columns(X(:, empty & isKept), deadKept);
                end
            end
            if isSpk
                run.step1a_fallback = r.signals(step1a_channels(r.signals, o));
                if o.DryRun   % a real run records info.P from the call itself
                    run.spike_params = spike_param_record(night6_v2_params());
                end
            end
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
                    if drop_outputs(plan, o)   % mode (B): each variable at its own cut
                        run.recovery_trim = night6_recovery_trim_outputs(outDir, ...
                            run.outputs, r.consumers, plan.recoveryStart, ...
                            o.RS.outputTimes, plan.fs, 'SlowWaveRate', o.SW);
                    end
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
            [D, info] = process_dataset_v2(D, 'PlotMode', figs, 'NanPadMs', P.spikes.nanPadMs, ...
                                           'Step1aChannels', step1a_channels(r.signals, o));
            save_spikes_v2(fullfile(outDir, [label '_spikes_v2.mat']), D, info, r.signals);
            extra.n_rpeaks = info.n_rpeaks;
            extra.spike_check = info.spike_check;
            extra.spike_params = spike_param_record(info.P);   % what her steps were given
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
            % RULING 2026-10-09 (c) 6: the run's masked spans go in as blankIdx too, at
            % the declared rate (night6_call_slow_wave).
            label = sprintf('%s_swm_%s', base, r.maskSignal);
            m = plan.masks(strcmp({plan.masks.consumer}, 'slow_wave') ...
                           & strcmp({plan.masks.signal}, r.maskSignal));
            if numel(m) ~= 1
                error('night6:mask', 'expected one slow_wave mask for %s, found %d', ...
                      r.maskSignal, numel(m));
            end
            S = night6_call_slow_wave(X, fs, P.slow_wave, outDir, label, base, r.signals, ...
                r.keep, r.maskSignal, m.spans, o.SlowWaveRate, figs);
            extra.slow_wave = S.slow_wave;
            extra.blank_idx = S.blank_idx;
            extra.slow_wave_rate = S.slow_wave_rate;
            extra.slow_wave_caveats = S.caveats;   % (h) 2, (f) 1
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
    % The spike P is NOT written here as text: night6_v2_params builds it (Andrea
    % 2026-10-09) and asserts threshSigma and the band; the values actually used are in
    % runs{}.spike_params, read from the struct (info.P after a real call).
    P.spikes = struct('method', 'process_dataset_v2', 'steps', {night6_v2_steps()}, ...
                      'nanPadMs', 5, ...
                      'source', ['Andrea 2026-10-09: P = night6_v2_params() (her ' ...
                                 'pipeline_params, bandpass 300-3000 Hz); values used: ' ...
                                 'runs{}.spike_params; her process_dataset steps minus ' ...
                                 'step1a and step1b; D from her bulk_load_one with D.rpeakSamples']);
    P.hr = struct('cutoff', 8, 'order', 4, 'edgeBufferSec', 0.75, 'winSec', 20, ...
                  'stepSec', 1, 'hrBrWinSec', 60, 'chanidx', 1, ...
                  'source', 'batch_process.m P.hr_* (= run_continuous.m, = T)');
    P.slow_wave = night6_slow_wave_settings();   % one site: the call and its caveats use it
    P.mmc = struct('gastricCols', [1 2 3], 'rpeak_source', ...
                   ['epoch beats file: rpeakVar heartlocs, rpeakUnits samples, rpeakFs fs ' ...
                    '(night6_mmc_opts, unit checked)'], 'other', 'extract_mmc defaults');
end

function F = function_provenance()
% Andrea's functions must resolve to processing_new - a stray copy would shadow them -
% and the wrapper's own (process_dataset_v2 and its step list) to this folder.
    F = struct();
    C = night6_calls();
    ours = {'process_dataset_v2', 'night6_v2_steps', 'night6_v2_params', ...
            'night6_step1a_fallback', 'night6_sha256_file', 'night6_recovery_start', ...
            'night6_recovery_lead_in', 'night6_recovery_trim_outputs', ...
            'night6_trim_modes', 'night6_check_trim_mode', 'night6_edge_settling', ...
            'night6_slow_wave_rates', 'night6_check_decimation', 'night6_decimate_masked', ...
            'night6_call_slow_wave', 'night6_keep_slow_wave', 'night6_slow_wave_settings', ...
            'night6_slow_wave_caveats', 'night6_pilot_root', 'night6_hr_outputs'};
    hers = [setdiff({C.name}, ours, 'stable'), night6_v2_steps(), ...
            {'step1a_blank_cardiac', 'pipeline_params', 'bulk_load_one'}];   % step1a: (j) 1 fallback
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
        F.(name{1}) = struct('path', p, 'sha256', night6_sha256_file(p));
    end
    % Not called, but params() copies its constants (HR, slow wave): its hash says
    % which version of the driver the parameters were taken from.
    p = which('batch_process');
    F.batch_process = struct('path', p, 'sha256', night6_sha256_file(p), 'role', 'parameter source');
end

function s = spike_param_record(P)
% The spike parameters as USED, copied from the struct handed to her steps (never
% literal text): detection threshold, band, filter, guards, refractory, polarity.
    s = struct();
    for f = {'threshSigma', 'bandpassLow', 'bandpassHigh', 'filterOrder', ...
             'envCardiacGuardMs', 'edgeBufferMs', 'refractoryMs', 'detectPolarity'}
        if ~isfield(P, f{1})
            error('night6:spikeParams', 'P has no %s: cannot record what was used', f{1});
        end
        s.(f{1}) = P.(f{1});
    end
    s.source = 'night6_v2_params (pipeline_params + 300-3000 Hz), as handed to her steps';
    % RULING 2026-10-09 (c) 1: where edgeBufferMs came from - the declaration, its hash, the
    % measured settling it covers and the measurement files' SHA-256.
    s.edgeBufferMs_source = edge_settling_record(night6_edge_settling(), 'spikes');
end

function r = edge_settling_record(E, consumer)
% One consumer's measured edge settling as applied, with the declaration it was read from.
    r = E.(consumer);
    r.declaration = E.file;
    r.declaration_sha256 = E.sha256;
end

function tf = step1a_channels(signals, o)
% Which spike signals (<cuff>_T) belong to an animal x cuff on the declared fallback
% list (ruling (j) 1). Any other spike signal name is refused rather than guessed.
    tf = false(1, numel(signals));
    for j = 1:numel(signals)
        tok = regexp(signals{j}, '^([A-Za-z]+)_T$', 'tokens', 'once');
        if isempty(tok)
            error('night6:fallbackSignal', 'spike signal %s is not <cuff>_T', signals{j});
        end
        tf(j) = any(strcmp(o.fallback.keys, sprintf('%s|%s', o.animal, tok{1})));
    end
end

function [dead, why] = dead_columns(X, signals)
% Columns with no finite sample, or whose finite samples are all equal (an exactly-zero
% column included; invariant 41), for every call - and one sentence naming them.
    noValid = all(isnan(X), 1);
    hi = max(X, [], 1, 'omitnan');
    lo = min(X, [], 1, 'omitnan');
    constant = ~noValid & hi == lo;
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

function tol = tolerance_stand_in(prov)
% The STAND-IN tolerance label a mask provenance carries (extra.TOLERANCE_STAND_IN, written
% by night4 on every mask made on that table, RULING 2026-10-09 (g) 7), or ''. A present but
% empty mark still counts.
    tol = '';
    if isfield(prov, 'extra') && isstruct(prov.extra) ...
            && isfield(prov.extra, 'TOLERANCE_STAND_IN')
        tol = char(string(prov.extra.TOLERANCE_STAND_IN));
        if isempty(tol), tol = 'TOLERANCE_STAND_IN (no label)'; end
    end
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

function tf = drop_outputs(plan, o)
% True when this epoch's outputs are cut per variable (mode (B), RULING 2026-10-09 item
% 6, on a stim_recovery epoch the starts file applies to).
    tf = strcmp(o.RecoveryTrimMode, 'mask_to_electrical_drop_outputs') ...
        && ~isempty(plan.recoveryStart) && plan.recoveryStart.applies;
end

function tf = same_recovery_start(R, RS, mode)
% Resume only a stim_recovery epoch made with THIS recovery-starts file and THIS trim
% mode: a changed file or mode (or none recorded) reruns it, never reports it done with
% another set of starts or another semantics.
    tf = true;
    applies = isfield(R, 'recovery_start') && isstruct(R.recovery_start) ...
        && isfield(R.recovery_start, 'applies') && R.recovery_start.applies;
    if applies
        tf = ~isempty(RS) && isfield(R.recovery_start, 'sha256') ...
            && strcmp(R.recovery_start.sha256, RS.sha256) ...
            && isfield(R, 'recovery_trim_mode') && strcmp(R.recovery_trim_mode, mode);
    elseif isfield(R, 'condition') && strcmp(R.condition, 'stim_recovery')
        tf = false;
    end
end

function o = ternary(c, a, b)
    if c, o = a; else, o = b; end
end
