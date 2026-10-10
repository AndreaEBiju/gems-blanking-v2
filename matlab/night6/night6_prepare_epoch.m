function plan = night6_prepare_epoch(M, fileLabels, metaChannels, nFile, fsFile, units, beats, condition, varargin)
% NIGHT6_PREPARE_EPOCH  Plan one epoch of a Night 6 run from its mask file. No arrays.
%
%   plan = night6_prepare_epoch(M, fileLabels, metaChannels, nFile, fsFile, units, beats, condition)
%   plan = night6_prepare_epoch(..., condition, 'Recovery', ...
%                               struct('starts', RS, 'session', s, 'mode', trimMode))
%
%   M             load() of one e<start>_masks.mat (gems_blanking_v2.emit.handoff)
%   fileLabels    the recording file's own chanlabels, in column order (authoritative
%                 for ORDER)
%   metaChannels  meta.json "channels" (jsondecode: struct array or cell of structs);
%                 authoritative for geometry (cuff_id, contact_index, role)
%   nFile, fsFile samples and sample rate of the recording file
%   units         'V' | 'mV' | 'uV' - declared, never inferred (invariant 14).
%                 processing_new works in VOLTS, so the plan carries the scale to volts.
%   beats         the recording's beat train, or [] when it has none: a struct with
%                   data    load() of the train file (heartlocs 1-based from its ORIGIN)
%                   record  the mask provenance's extra.beats_file (jsondecode), written
%                           by the one train resolver: origin_sample0 (0-based file
%                           sample of heartloc 1), sha256, n_in_epoch
%                   sha256  the train file's SHA-256 as read here
%   condition     the recording's condition from the mask provenance ('baseline',
%                 'stim_recovery', 'pre'), or '' when the provenance names none
%
% The epoch is samples i0+1 .. i0+n of the file (1-based), with i0 = epochStartSample0,
% the exact 0-based start sample the mask writer sliced with, and n = nSamples
% (Andrea, 2026-10-09: the mask file carries the sample and MATLAB never converts
% seconds to samples - invariants 15, 22). epochStart_s is carried into the record only.
% Every blank_<consumer>_<token> span is 1-based inclusive into the EPOCH (task 15), so
% it lands on epoch rows sp(k,1):sp(k,2) unchanged.
%
% plan.runs lists the calls to make, in night6_calls() order. A run serves consumers
% that share one call AND one mask; hrv and breathing get two HR runs when their masks
% differ (invariant 2: a mask is never merged across consumers). Consumers in
% notcomputed_json are skipped with their reason (RULING 2026-10-08 (f) 6); mmc,
% which needs R-peaks, is skipped when the recording has no beats in the epoch - and
% on a 'pre' recording with no beat train at all it is "not computed" (Andrea,
% 2026-10-09: pre files were often recorded with the laptop charger plugged in).
%
% slow_wave runs ONE ANT CHANNEL AT A TIME (Andrea, 2026-10-09): for channel i the run
% applies channel i's mask to all three ANT columns (run.maskSignal), so
% slowWaveAnalysis_new's joint any(isnan) mask equals mask i, and only channel i's
% outputs are kept (run.keep). Channels whose masks are identical get bit-identical
% inputs, so they share one call and each keeps its own column - the outputs are those
% of separate calls exactly, at a third of the cost.
%
% PERI-R (RULING 2026-10-08 (k) 1; review fix 2, 2026-10-09, invariant 28): when the spike
% consumer reads any signal, the file must carry its peri-R record - perir_json, a
% perir_spikes_<token> per spike signal, and the provenance's spike_peri_r - or the epoch is
% refused (night6:periR). emit.handoff writes that record for EVERY such file since
% f15bf43: with the train's provenance (sha256, origin) when the routing gives a train, and
% train = "none: ..." with no spans when it gives none. So "no spans because no train" is
% recorded, and a file with no perir_json was written before (k) 1 - never read as "no
% train". plan.periR states which (train_state 'train' | 'none'); [] when no spike signal.
%
% NO HEARTBEAT REFERENCE (RULING 2026-10-09 (g) 1; review 2026-10-10 fix 4): when the spike
% consumer reads any signal AND the recording has a train (peri-R train_state 'train', or a
% beats file), the file must carry noheartref_json (rule, signals = exactly the spike
% signals, minutes), a noheartref_spikes_<token> per spike signal and the provenance's
% spike_no_heartbeat_reference - or the epoch is refused by name (night6:noHeartRef). A file
% without them was written before (g) 1, when its no-beat minutes were BLANKED under (b) 2
% rather than kept and flagged, so it is not the mask (g) 1 rules. plan.noHeartRef is the
% record (rule, minute count); [] when the check does not apply.
%
% RECOVERY START (RULING 2026-10-08 (k) 2), name-value 'Recovery', struct(starts, session,
% mode) - mode is the declared trim mode (night6_trim_modes), refused by name if unknown
% or withdrawn:
% starts is night6_recovery_start(file) or []. For a stim_recovery epoch every consumer that
% runs must have its start there, or the epoch is refused by name; when the file's
% electrical settling (mode (B), RULING 2026-10-09 item 6) is later than the epoch start,
% every consumer gets its leading rows masked (NaN, like any motion
% span) BEFORE the runs are planned, so calls that share a mask still share it exactly
% (night6_recovery_lead_in). plan.recoveryStart is the record; [] when 'Recovery' is not
% given (test harnesses that plan an epoch only - night6_run_recording always gives it).
    ip = inputParser;
    ip.addParameter('Recovery', [], @(x) isempty(x) || (isstruct(x) && isscalar(x) ...
                    && all(isfield(x, {'starts', 'session', 'mode'}))));
    ip.parse(varargin{:});
    recovery = ip.Results.Recovery;
    known = {'spikes', 'slow_wave', 'mmc', 'hrv', 'breathing', 'velocity'};
    if ~ismember(units, {'V', 'mV', 'uV'})
        error('night6:units', 'units must be declared as V, mV or uV; got ''%s''', char(units));
    end
    scale = struct('V', 1, 'mV', 1e-3, 'uV', 1e-6);

    plan = struct();
    plan.fs = double(M.fs);
    plan.n = double(M.nSamples);
    plan.epochStart_s = double(M.epochStart_s);   % for the record only, never for indexing
    if ~isfield(M, 'epochStartSample0')
        error('night6:startSample', ['the mask file has no epochStartSample0 (the exact ' ...
              'epoch start sample, written by emit.handoff since 2026-10-09); MATLAB does ' ...
              'not derive it from epochStart_s']);
    end
    plan.i0 = double(M.epochStartSample0);
    if ~isscalar(plan.i0) || plan.i0 ~= fix(plan.i0)
        error('night6:startSample', 'epochStartSample0 must be one integer sample index');
    end
    plan.condition = char(condition);
    plan.units = char(units);
    plan.scaleToVolts = scale.(char(units));
    if abs(double(fsFile) - plan.fs) > 1e-9
        error('night6:fs', 'mask file fs %.6f, recording fs %.6f', plan.fs, double(fsFile));
    end
    if plan.n ~= round(plan.n) || plan.n < 1 || plan.i0 < 0 || plan.i0 + plan.n > nFile
        error('night6:epoch', ['epoch [%d, %d] (0-based start, length) does not fit a ' ...
              'recording of %d samples'], plan.i0, plan.n, nFile);
    end

    channels = normalise_channels(metaChannels);
    fileLabels = cellstr(fileLabels);
    metaLabels = cellfun(@(c) char(c.label), channels, 'UniformOutput', false);
    if ~isequal(metaLabels(:)', fileLabels(:)')
        error('night6:labels', 'meta.json channels [%s] do not match the file''s chanlabels [%s]', ...
              strjoin(metaLabels, ' '), strjoin(fileLabels, ' '));
    end

    % ---- masks: one entry per (consumer, signal), never merged ------------------
    plan.notComputed = decode_object(M, 'notcomputed_json');
    masks = struct('consumer', {}, 'signal', {}, 'token', {}, 'spans', {});
    names = fieldnames(M);
    for i = 1:numel(names)
        nm = names{i};
        if ~startsWith(nm, 'blank_'), continue, end
        hit = known(cellfun(@(c) startsWith(nm, ['blank_' c '_']), known));
        if numel(hit) ~= 1
            error('night6:maskName', '%s does not name exactly one known consumer', nm);
        end
        token = nm(numel(['blank_' hit{1} '_']) + 1:end);
        sp = check_spans(M.(nm), plan.n, nm);
        masks(end + 1) = struct('consumer', hit{1}, 'signal', night6_signal_from_token(token), ...
                                'token', token, 'spans', sp); %#ok<AGROW>
    end
    plan.masks = masks;
    plan.periR = check_peri_r(M, masks, plan.n);
    plan.noHeartRef = check_no_heartref(M, masks, plan.periR, beats, plan.n);
    plan.notMeasuredMmc = struct();
    for i = 1:numel(names)
        if startsWith(names{i}, 'notmeasured_mmc_')
            sig = night6_signal_from_token(names{i}(numel('notmeasured_mmc_') + 1:end));
            plan.notMeasuredMmc.(matlab.lang.makeValidName(sig)) = ...
                check_spans(M.(names{i}), plan.n, names{i});
        end
    end

    % ---- signal recipes: x = Y(:, cols) * weights * scaleToVolts ---------------
    plan.recipes = struct();
    for k = 1:numel(masks)
        key = matlab.lang.makeValidName(masks(k).signal);
        r = recipe(masks(k).signal, fileLabels, channels);
        if isfield(plan.recipes, key) && ~isequal(plan.recipes.(key), r)
            error('night6:recipeKey', 'two signals share the recipe key %s (invariant 27)', key);
        end
        plan.recipes.(key) = r;
    end

    % ---- the epoch's beats (whole-file beats re-based to the epoch) ------------
    plan.beats = slice_beats(beats, plan);

    % ---- consumer status, then runs ---------------------------------------------
    plan.consumers = struct();
    for c = known
        cn = c{1};
        sel = strcmp({masks.consumer}, cn);
        st = struct('status', '', 'reason', '', 'signals', {{masks(sel).signal}});
        nc = isfield(plan.notComputed, cn);
        if nc && any(sel)
            error('night6:notComputedReads', ...
                  '%s is marked not computed but the file carries masks for it', cn);
        end
        if nc
            st.status = 'skipped_not_computed';
            st.reason = char(plan.notComputed.(cn));
        elseif ~any(sel)
            st.status = 'reads_nothing';
            st.reason = 'no mask in this file: the consumer reads no signal in this recording';
        elseif strcmp(cn, 'velocity')
            st.status = 'skipped_out_of_build';
            st.reason = 'task 18 (conduction velocity) is out of this build (RULING 2026-10-07 (b) R5)';
        else
            st.status = 'to_run';
        end
        plan.consumers.(cn) = st;
    end
    gastric = stomach_labels(channels);
    for cn = {'slow_wave', 'mmc'}
        st = plan.consumers.(cn{1});
        if strcmp(st.status, 'to_run') && ~isequal(sort(st.signals), sort(gastric))
            error('night6:gastric', '%s reads [%s]; Andrea''s call takes exactly [%s]', ...
                  cn{1}, strjoin(st.signals, ' '), strjoin(gastric, ' '));
        end
    end
    for cn = {'hrv', 'breathing'}
        st = plan.consumers.(cn{1});
        if ~strcmp(st.status, 'to_run'), continue, end
        if numel(st.signals) ~= 1
            error('night6:hrSignals', '%s reads %d signals; the HR call takes one', ...
                  cn{1}, numel(st.signals));
        end
        if isempty(plan.beats)
            error('night6:noBeats', ['%s is to be computed but the recording has no beats ' ...
                  'file: the mask file and the store disagree'], cn{1});
        end
        if ~strcmp(st.signals{1}, plan.beats.beatChannel)
            error('night6:beatChannel', '%s reads %s but the beats were found on %s', ...
                  cn{1}, st.signals{1}, plan.beats.beatChannel);
        end
        if isempty(plan.beats.heartlocs)
            % Invariant 41: the HR call on an empty beat train would answer from nothing.
            plan.consumers.(cn{1}).status = 'skipped_no_beats_in_epoch';
            plan.consumers.(cn{1}).reason = sprintf(['the beats file has %d beats, none ' ...
                'inside this epoch: HR_BR_HRVAnalysis_beats is not called on an empty ' ...
                'train'], plan.beats.nWholeFile);
        end
    end
    % RULING 2026-10-08 (j) 4 ((h) 7): "pre" files skip mmc, marked not computed. Only a
    % pre file WITH NO BEAT TRAIN takes this branch; a pre file with beats runs mmc, and
    % a non-pre file without beats is skipped_no_rpeaks below (diagnosed, not excused).
    if strcmp(plan.consumers.mmc.status, 'to_run') && isempty(plan.beats) ...
            && strcmp(plan.condition, 'pre')
        plan.consumers.mmc.status = 'skipped_not_computed';
        plan.consumers.mmc.reason = ['no count-gated beat train on a "pre" recording: mmc ' ...
            'is not computed (Andrea, 2026-10-09: pre files were often recorded with the ' ...
            'laptop charger plugged in, which adds line noise)'];
    elseif strcmp(plan.consumers.mmc.status, 'to_run') && ...
            (isempty(plan.beats) || isempty(plan.beats.heartlocs))
        plan.consumers.mmc.status = 'skipped_no_rpeaks';
        plan.consumers.mmc.reason = ['extract_mmc needs R-peaks for its cardiac blanking ' ...
            'and this epoch has no stored beats (no beats file, or none in the epoch)'];
    end

    plan.recoveryStart = [];
    if ~isempty(recovery)
        toRun = known(cellfun(@(c) strcmp(plan.consumers.(c).status, 'to_run'), known));
        [lead, plan.recoveryStart] = night6_recovery_lead_in(recovery.starts, ...
            char(recovery.session), plan.condition, plan.i0, plan.n, plan.fs, toRun, ...
            recovery.mode);
        masks = apply_lead_in(masks, lead);
        plan.masks = masks;
    end

    plan.runs = struct('call', {}, 'consumers', {}, 'signals', {}, 'maskSignal', {}, 'keep', {});
    for C = night6_calls()
        want = C.consumers(cellfun(@(c) strcmp(plan.consumers.(c).status, 'to_run'), C.consumers));
        if strcmp(C.name, 'slowWaveAnalysis_new') && ~isempty(want)
            plan.runs = [plan.runs, slow_wave_runs(masks, gastric)]; %#ok<AGROW>
            continue
        end
        while ~isempty(want)
            lead = want{1};
            same = cellfun(@(c) same_mask(masks, lead, c), want);
            sigs = plan.consumers.(lead).signals;
            if strcmp(C.name, 'extract_mmc')
                sigs = gastric;          % her column order: ANT1, ANT2, ANT3
            end
            plan.runs(end + 1) = struct('call', C.name, 'consumers', {want(same)}, ...
                                        'signals', {sigs}, 'maskSignal', '', ...
                                        'keep', {sigs}); %#ok<AGROW>
            want = want(~same);
        end
    end
end

% ==========================================================================
function masks = apply_lead_in(masks, lead)
% Add the span [1, lead.<consumer>] to every mask of that consumer and merge the spans
% (sorted, overlapping or adjacent runs joined), so two masks with the same NaN rows
% compare equal and still share one call.
    for k = 1:numel(masks)
        c = masks(k).consumer;
        if ~isfield(lead, c) || lead.(c) < 1, continue, end
        masks(k).spans = merge_spans([1, lead.(c); masks(k).spans]);
    end
end

function s = merge_spans(s)
    s = sortrows(s, 1);
    out = s(1, :);
    for k = 2:size(s, 1)
        if s(k, 1) <= out(end, 2) + 1
            out(end, 2) = max(out(end, 2), s(k, 2));
        else
            out(end + 1, :) = s(k, :); %#ok<AGROW>
        end
    end
    s = out;
end

% ==========================================================================
function runs = slow_wave_runs(masks, gastric)
% One slowWaveAnalysis_new run per distinct slow_wave mask (Andrea, 2026-10-09). Run k
% applies the mask of its first channel to ALL THREE ANT columns (maskSignal) and keeps
% the outputs of every channel whose own mask is identical (keep) - their inputs are
% bit-identical, so one call gives each of them exactly what its own call would.
    runs = struct('call', {}, 'consumers', {}, 'signals', {}, 'maskSignal', {}, 'keep', {});
    sw = masks(strcmp({masks.consumer}, 'slow_wave'));
    todo = gastric;
    while ~isempty(todo)
        lead = sw(strcmp({sw.signal}, todo{1}));
        if numel(lead) ~= 1
            error('night6:slowWaveMask', 'expected one slow_wave mask for %s, found %d', ...
                  todo{1}, numel(lead));
        end
        same = cellfun(@(s) isequal(sw(strcmp({sw.signal}, s)).spans, lead.spans), todo);
        runs(end + 1) = struct('call', 'slowWaveAnalysis_new', 'consumers', {{'slow_wave'}}, ...
                               'signals', {gastric}, 'maskSignal', todo{1}, ...
                               'keep', {todo(same)}); %#ok<AGROW>
        todo = todo(~same);
    end
end

% ==========================================================================
function sp = check_spans(v, n, what)
    sp = double(v);
    if isempty(sp), sp = zeros(0, 2); return, end
    if size(sp, 2) ~= 2 || any(sp(:) ~= round(sp(:))) || any(sp(:) < 1) ...
            || any(sp(:) > n) || any(sp(:, 2) < sp(:, 1))
        error('night6:spans', ['%s: spans must be N x 2 1-based inclusive sample indices ' ...
              'into the epoch, 1 <= start <= stop <= %d'], what, n);
    end
end

function S = decode_object(M, field)
    if ~isfield(M, field)
        error('night6:missing', 'the mask file has no %s (written by handoff since a2e91d6)', field);
    end
    S = jsondecode(char(M.(field)));
    if isempty(S), S = struct(); end
end

function ch = normalise_channels(c)
    if isstruct(c), ch = num2cell(c(:)'); else, ch = c(:)'; end
end

function g = stomach_labels(channels)
    isStomach = cellfun(@(c) isfield(c, 'role') && strcmp(c.role, 'stomach'), channels);
    g = cellfun(@(c) char(c.label), channels(isStomach), 'UniformOutput', false);
end

function r = recipe(sig, labels, channels)
% How a consumer's signal is formed from the file's columns (in the file's units).
    col = @(lab) find_one(labels, lab, sig);
    tok = regexp(sig, '^([A-Za-z]+)_T$', 'tokens', 'once');
    if contains(sig, '-')
        parts = strsplit(sig, '-');
        r = struct('kind', 'pair', 'cols', [col(parts{1}) col(parts{2})], 'weights', [1; -1]);
    elseif ~isempty(tok)
        % Software tripole, task 04: T = a*V1 + b*V3 - V2 with a = b = 0.5 (applied
        % weights are naive by ruling; never fitted). Contacts from meta.json.
        cuff = tok{1};
        cols = zeros(1, 3);
        for k = 1:3
            hit = cellfun(@(c) isfield(c, 'cuff_id') && strcmp(c.cuff_id, cuff) && ...
                isfield(c, 'contact_index') && c.contact_index == k && ...
                strcmp(c.config, 'independent'), channels);
            if nnz(hit) ~= 1
                error('night6:tripole', '%s: cuff %s needs exactly one independent contact %d', ...
                      sig, cuff, k);
            end
            cols(k) = col(char(channels{hit}.label));
        end
        r = struct('kind', 'tripole', 'cols', cols, 'weights', [0.5; -1; 0.5]);
    else
        r = struct('kind', 'raw', 'cols', col(sig), 'weights', 1);
    end
end

function i = find_one(labels, lab, sig)
    i = find(strcmp(labels, lab));
    if numel(i) ~= 1
        error('night6:label', 'signal %s needs channel %s, found %d in the file', sig, lab, numel(i));
    end
end

function P = check_peri_r(M, masks, n)
% Review fix 2: a spike consumer that reads signals needs the file's peri-R record.
    P = [];
    sel = strcmp({masks.consumer}, 'spikes');
    if ~any(sel), return, end
    sigs = sort({masks(sel).signal});
    if ~isfield(M, 'perir_json')
        error('night6:periR', ['the spike consumer reads [%s] but the mask file has no ' ...
              'perir_json: it was written before RULING 2026-10-08 (k) 1 and carries no ' ...
              'peri-R spans. A spike mask without its peri-R record is refused; re-emit ' ...
              'the masks'], strjoin(sigs, ' '));
    end
    J = jsondecode(char(M.perir_json));
    if ~isfield(J, 'train') || ~isfield(J, 'signals') || ~isfield(J, 'n_spans') ...
            || ~isfield(J, 'window') || ~isfield(J.window, 'sha256')
        error('night6:periR', 'perir_json lacks train, signals, n_spans or window.sha256');
    end
    if ischar(J.train) && startsWith(J.train, 'none:')
        state = 'none';
        if J.n_spans ~= 0
            error('night6:periR', 'perir_json says no train but carries %d spans', J.n_spans);
        end
    elseif isstruct(J.train) && isfield(J.train, 'sha256') && ischar(J.train.sha256) ...
            && numel(J.train.sha256) == 64
        state = 'train';
    else
        error('night6:periR', ['perir_json train is neither a train record (with its ' ...
              'sha256) nor "none: ..." - the train state is unknown']);
    end
    got = cellstr(J.signals);
    got = sort(got(:))';
    if ~isequal(got, sigs(:)')
        error('night6:periR', 'perir_json covers [%s]; the spike consumer reads [%s]', ...
              strjoin(got, ' '), strjoin(sigs, ' '));
    end
    for t = {masks(sel).token}
        nm = ['perir_spikes_' t{1}];
        if ~isfield(M, nm)
            error('night6:periR', 'the mask file has perir_json but no %s', nm);
        end
        sp = check_spans(M.(nm), n, nm);
        if size(sp, 1) ~= J.n_spans
            error('night6:periR', '%s has %d spans, perir_json %d', nm, size(sp, 1), J.n_spans);
        end
    end
    if ~isfield(M, 'provenance_json') ...
            || ~isfield(jsondecode(char(M.provenance_json)), 'spike_peri_r')
        error('night6:periR', 'the mask provenance names no spike_peri_r record');
    end
    P = struct('train_state', state, 'n_spans', J.n_spans, ...
               'window_sha256', J.window.sha256, 'signals', {got});
end

function H = check_no_heartref(M, masks, periR, beats, n)
% Review 2026-10-10 fix 4: no pre-(g) 1 mask file reaches the spike consumer when a train exists.
    H = [];
    sel = strcmp({masks.consumer}, 'spikes');
    if ~any(sel), return, end
    hasTrain = (~isempty(periR) && strcmp(periR.train_state, 'train')) || ~isempty(beats);
    if ~hasTrain, return, end
    sigs = sort({masks(sel).signal});
    if ~isfield(M, 'noheartref_json')
        error('night6:noHeartRef', ['the spike consumer reads [%s] and the recording has a ' ...
              'train, but the mask file has no noheartref_json: it was written before RULING ' ...
              '2026-10-09 (g) 1, so its no-beat minutes were blanked under (b) 2, not kept and ' ...
              'flagged. Refused; re-emit the masks'], strjoin(sigs, ' '));
    end
    J = jsondecode(char(M.noheartref_json));
    if ~isstruct(J) || ~all(isfield(J, {'rule', 'signals', 'minutes'}))
        error('night6:noHeartRef', 'noheartref_json lacks rule, signals or minutes');
    end
    got = sort(cellstr(J.signals));
    if ~isequal(got(:)', sigs(:)')
        error('night6:noHeartRef', 'noheartref_json covers [%s]; the spike consumer reads [%s]', ...
              strjoin(got, ' '), strjoin(sigs, ' '));
    end
    for t = {masks(sel).token}
        nm = ['noheartref_spikes_' t{1}];
        if ~isfield(M, nm)
            error('night6:noHeartRef', 'the mask file has noheartref_json but no %s', nm);
        end
        check_spans(M.(nm), n, nm);
    end
    if ~isfield(M, 'provenance_json') ...
            || ~isfield(jsondecode(char(M.provenance_json)), 'spike_no_heartbeat_reference')
        error('night6:noHeartRef', 'the mask provenance names no spike_no_heartbeat_reference');
    end
    H = struct('rule', char(J.rule), 'n_minutes', numel(J.minutes), 'signals', {got(:)'});
end

function tf = same_mask(masks, a, b)
    A = masks(strcmp({masks.consumer}, a));
    B = masks(strcmp({masks.consumer}, b));
    tf = numel(A) == numel(B) && isequal(sort({A.signal}), sort({B.signal}));
    if ~tf, return, end
    for k = 1:numel(A)
        tf = tf && isequal(A(k).spans, B(strcmp({B.signal}, A(k).signal)).spans);
    end
end

function E = slice_beats(T, plan)
% The recording's beat train -> the epoch: heartlocs in the epoch re-based to it,
% gapAfter kept with its beat, blankSpans clipped to the epoch and re-based (the
% trim_blank rule of batch_process.m).
%
% THE ORIGIN (fix (c), origin_check/REPORT.md, 2026-10-09): a stored heartloc h is
% 1-based into the routing REGION the beats were detected on, not into the file. The
% stim_rec trains' region starts at round(132 fs) while the file still declares
% epochStart_s = 0 (the routing readers rely on that 0), so the file's epochStart_s is
% NOT the origin and is never read here. The origin comes from the mask provenance's
% record (T.record.origin_sample0, the resolver's 0-based file sample of heartloc 1):
%     file sample (0-based)  f   = origin_sample0 + h - 1
%     epoch row   (1-based)  row = f - i0 + 1
% (never seconds x fs - invariant 15). Refused by name: a record without the origin, a
% train whose sha256 is not the record's, a file whose own epochStartSample0 disagrees
% with the record, and a beat count in the epoch other than the one Python computed.
    if isempty(T), E = []; return, end
    if ~isstruct(T) || ~all(isfield(T, {'data', 'record', 'sha256'}))
        error('night6:beatsArg', 'beats must be struct(data, record, sha256) or []');
    end
    B = T.data;
    ref = T.record;
    if ~isfield(ref, 'origin_sample0')
        error('night6:beatsOrigin', ['the mask provenance''s beats_file record has no ' ...
              'origin_sample0: the train''s origin is unknown, so its beats are not placed']);
    end
    off = double(ref.origin_sample0);
    if ~isscalar(off) || off ~= fix(off) || off < 0
        error('night6:beatsOrigin', 'beats_file origin_sample0 must be one integer >= 0');
    end
    if ~isfield(ref, 'sha256') || ~strcmpi(char(ref.sha256), char(T.sha256))
        error('night6:beatsSha', ['the beat train''s sha256 %s is not the one the mask ' ...
              'provenance records: the file changed since the masks were made'], char(T.sha256));
    end
    if ~isfield(ref, 'n_in_epoch')
        error('night6:beatsCount', 'the beats_file record has no n_in_epoch to reconcile against');
    end
    if abs(double(B.fs) - plan.fs) > 1e-9
        error('night6:beatsFs', 'beats fs %.6f, recording fs %.6f', double(B.fs), plan.fs);
    end
    if isfield(B, 'epochStartSample0') && double(B.epochStartSample0) ~= off
        error('night6:beatsStart', ['the beats file declares epochStartSample0 %d but the ' ...
              'mask provenance records origin_sample0 %d'], double(B.epochStartSample0), off);
    end
    h = double(B.heartlocs(:));
    fileSample0 = off + h - 1;
    row = fileSample0 - plan.i0 + 1;
    keep = row >= 1 & row <= plan.n;
    if nnz(keep) ~= double(ref.n_in_epoch)
        error('night6:beatsCount', ['%d beats fall in this epoch here, %d by the Python ' ...
              'side''s count (n_in_epoch): the two disagree on the origin or the epoch'], ...
              nnz(keep), double(ref.n_in_epoch));
    end
    E = struct('heartlocs', row(keep), 'beatChannel', char(B.beatChannel), ...
               'gapAfter', false(nnz(keep), 1), 'blankSpans', zeros(0, 2), ...
               'nWholeFile', numel(h), 'originSample0', off, 'sha256', char(T.sha256));
    if isfield(B, 'gapAfter') && ~isempty(B.gapAfter)
        g = logical(B.gapAfter(:));
        if numel(g) ~= numel(h)
            error('night6:gapAfter', 'gapAfter has %d entries for %d heartlocs', numel(g), numel(h));
        end
        E.gapAfter = g(keep);
    end
    if isfield(B, 'blankSpans') && ~isempty(B.blankSpans)
        s = (off + double(B.blankSpans) - 1) - plan.i0 + 1;   % same frame as heartlocs
        s(s(:, 2) < 1 | s(:, 1) > plan.n, :) = [];
        s(:, 1) = max(s(:, 1), 1);
        s(:, 2) = min(s(:, 2), plan.n);
        E.blankSpans = s;
    end
end
