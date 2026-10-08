function plan = night6_prepare_epoch(M, fileLabels, metaChannels, nFile, fsFile, units, beats)
% NIGHT6_PREPARE_EPOCH  Plan one epoch of a Night 6 run from its mask file. No arrays.
%
%   plan = night6_prepare_epoch(M, fileLabels, metaChannels, nFile, fsFile, units, beats)
%
%   M             load() of one e<start>_masks.mat (gems_blanking_v2.emit.handoff)
%   fileLabels    the recording file's own chanlabels, in column order (authoritative
%                 for ORDER)
%   metaChannels  meta.json "channels" (jsondecode: struct array or cell of structs);
%                 authoritative for geometry (cuff_id, contact_index, role)
%   nFile, fsFile samples and sample rate of the recording file
%   units         'V' | 'mV' | 'uV' - declared, never inferred (invariant 14).
%                 processing_new works in VOLTS, so the plan carries the scale to volts.
%   beats         the recording's whole-file beats file (load()), or [] when it has none
%
% The epoch is samples i0+1 .. i0+n of the file (1-based), with
% i0 = round_half_even(epochStart_s * fs) - Python's round, the rule the mask writer
% used - and n = nSamples. Every blank_<consumer>_<token> span is 1-based inclusive
% into the EPOCH (task 15), so it lands on epoch rows sp(k,1):sp(k,2) unchanged.
%
% plan.runs lists the calls to make, in night6_calls() order. A run serves consumers
% that share one call AND one mask; hrv and breathing get two HR runs when their masks
% differ (invariant 2: a mask is never merged across consumers). Consumers in
% notcomputed_json are skipped with their reason (RULING 2026-10-08 (f) 6); mmc,
% which needs R-peaks, is skipped when the recording has no beats in the epoch.
    known = {'spikes', 'slow_wave', 'mmc', 'hrv', 'breathing', 'velocity'};
    if ~ismember(units, {'V', 'mV', 'uV'})
        error('night6:units', 'units must be declared as V, mV or uV; got ''%s''', char(units));
    end
    scale = struct('V', 1, 'mV', 1e-3, 'uV', 1e-6);

    plan = struct();
    plan.fs = double(M.fs);
    plan.n = double(M.nSamples);
    plan.epochStart_s = double(M.epochStart_s);
    plan.i0 = night6_round_half_even(plan.epochStart_s * plan.fs);
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
    if strcmp(plan.consumers.mmc.status, 'to_run') && ...
            (isempty(plan.beats) || isempty(plan.beats.heartlocs))
        plan.consumers.mmc.status = 'skipped_no_rpeaks';
        plan.consumers.mmc.reason = ['extract_mmc needs R-peaks for its cardiac blanking ' ...
            'and this epoch has no stored beats (no beats file, or none in the epoch)'];
    end

    plan.runs = struct('call', {}, 'consumers', {}, 'signals', {});
    for C = night6_calls()
        want = C.consumers(cellfun(@(c) strcmp(plan.consumers.(c).status, 'to_run'), C.consumers));
        while ~isempty(want)
            lead = want{1};
            same = cellfun(@(c) same_mask(masks, lead, c), want);
            sigs = plan.consumers.(lead).signals;
            if strcmp(C.name, 'slowWaveAnalysis_new') || strcmp(C.name, 'extract_mmc')
                sigs = gastric;          % her column order: ANT1, ANT2, ANT3
            end
            plan.runs(end + 1) = struct('call', C.name, 'consumers', {want(same)}, ...
                                        'signals', {sigs}); %#ok<AGROW>
            want = want(~same);
        end
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

function tf = same_mask(masks, a, b)
    A = masks(strcmp({masks.consumer}, a));
    B = masks(strcmp({masks.consumer}, b));
    tf = numel(A) == numel(B) && isequal(sort({A.signal}), sort({B.signal}));
    if ~tf, return, end
    for k = 1:numel(A)
        tf = tf && isequal(A(k).spans, B(strcmp({B.signal}, A(k).signal)).spans);
    end
end

function E = slice_beats(B, plan)
% Whole-file beats -> the epoch: heartlocs (1-based) in (i0, i0+n] re-based by i0,
% gapAfter kept with its beat, blankSpans clipped to the epoch and re-based (the
% trim_blank rule of batch_process.m). The beats file's own epochStart_s is honoured.
    if isempty(B), E = []; return, end
    if abs(double(B.fs) - plan.fs) > 1e-9
        error('night6:beatsFs', 'beats fs %.6f, recording fs %.6f', double(B.fs), plan.fs);
    end
    off = 0;
    if isfield(B, 'epochStart_s')
        off = night6_round_half_even(double(B.epochStart_s) * plan.fs);
    end
    h = double(B.heartlocs(:)) + off - plan.i0;
    keep = h >= 1 & h <= plan.n;
    E = struct('heartlocs', h(keep), 'beatChannel', char(B.beatChannel), ...
               'gapAfter', false(nnz(keep), 1), 'blankSpans', zeros(0, 2), ...
               'nWholeFile', numel(h));
    if isfield(B, 'gapAfter') && ~isempty(B.gapAfter)
        g = logical(B.gapAfter(:));
        if numel(g) ~= numel(h)
            error('night6:gapAfter', 'gapAfter has %d entries for %d heartlocs', numel(g), numel(h));
        end
        E.gapAfter = g(keep);
    end
    if isfield(B, 'blankSpans') && ~isempty(B.blankSpans)
        s = double(B.blankSpans) + off - plan.i0;
        s(s(:, 2) < 1 | s(:, 1) > plan.n, :) = [];
        s(:, 1) = max(s(:, 1), 1);
        s(:, 2) = min(s(:, 2), plan.n);
        E.blankSpans = s;
    end
end
