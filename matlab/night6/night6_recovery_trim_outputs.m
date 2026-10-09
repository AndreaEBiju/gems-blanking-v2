function T = night6_recovery_trim_outputs(outDir, files, consumers, rec, OT, fs)
% NIGHT6_RECOVERY_TRIM_OUTPUTS  Trim every output variable at its own cut (mode (B)).
%
%   T = night6_recovery_trim_outputs(outDir, files, consumers, rec, OT, fs)
%
%   outDir     the epoch's output folder
%   files      the names of the files this run created (after night6_keep_slow_wave)
%   consumers  the run's consumers
%   rec        plan.recoveryStart (applies true): rec.cuts{} holds EVERY cut of the file's
%              row (night6_recovery_lead_in), rec.analyses.<analysis> every start
%   OT         RS.outputTimes: the output time map (Python extent.recovery_start.OUTPUT_VARS)
%   fs         the epoch's sample rate
%
% RULING 2026-10-09 item 6, mode (B). The input of every analysis was masked only through
% the electrical settling; here each output file the run kept is matched to its kind in
% the map, loaded, and:
%
% 1. CUT. Every variable the map classifies as 'trim' is cut at ITS OWN cut, named by the
%    map (owner.output.class): L = the cut's output_rows_before_start (epoch rows before
%    the cut point). Its class must be one of the map's trim classes - valid_only (i):
%    the cut is the electrical settling + its own input's settling; filled_or_filtered
%    (ii): the electrical settling + its full reach - or it is refused by name
%    ('night6:trimClass'); a cut the file's row does not hold is refused
%    ('night6:trimCut'). A byproduct (hrv's heart rate in a breathing-only run) is cut at
%    its own cut whoever ran. p = the stamp's 0-based epoch position by its convention
%    (row1, sec0, sec_row1, sec_xchan_delay); an entry is BEFORE the cut iff
%    p < L - 1e-6 (a stamp of an integer row computed in floating point counts as that
%    row). Actions:
%      nan         the value is set NaN (her own not-computed marker), the stamp is kept;
%      drop        the entry is removed from its list, with every list on the same stamps;
%      nan_events  a logical event series becomes DOUBLE (1 event, 0 none) and its rows
%                  before the cut are NaN: not computed, never "no event".
% 2. VALID FRACTION. Every windowed variable carries the fraction of its window's input
%    rows that were valid, over the window exactly as her code defines it, from the input
%    as the call saw it (the file before any cut): her own variable where she saves one
%    (kind 'her', recorded), otherwise a sibling <name>_validFraction of the same shape,
%    for every row of the stamp (computed or not).
% 3. RECOMPUTED AVERAGES. Every variable the map classifies as 'recomputed' (a
%    whole-epoch average or count of a trimmed series) is recomputed from the KEPT
%    values by her own expression; her original is kept in the marker.
%
% NOT COMPUTED TRAVELS WITH THE FILE. Every trimmed file gets one added top-level
% variable, named by the map (OT.markerVariable): per trimmed variable its owner, class,
% cut, the edge rule of her code, action, stamp and convention, the cut point (seconds and
% 0-based file sample), the first computed epoch row (1-based) and sample (0-based), the
% mode, per leaf the entries that are not computed (1-based inclusive runs; for 'drop' how
% many were removed) and its valid-fraction variable; the variables Night 6 added
% (added_variables); and every recomputed value with her original (recomputed). Her
% variables keep their names; types and shapes too, except the mmc events (double).
% A file that already holds the marker is refused ('night6:trimTwice').
%
% Every stamp and every input is read from the file as the call wrote it, before anything
% is changed, so co-indexed lists are cut by the same entries. Everything else is
% untouched: a value at or after its cut is bit-identical to the call's own output (the
% events as double). The file is rewritten atomically in its own MAT version.
%
% Never silent: an epoch-wide scalar is listed (computed over [electrical settling, epoch
% end]); a variable the map marks 'unknown', or that the map does not list, is left
% UNTRIMMED and listed by name (in the record and in the marker); a file matching no kind
% (a figure) is listed untrimmed; a stamp or a window whose length does not match its
% value is refused ('night6:trimShape').
    T = struct('mode', 'mask_to_electrical_drop_outputs', 'ruling', char(OT.ruling), ...
               'files', {{}}, 'untrimmed_files', {{}});
    if ~isfield(rec, 'cuts') || ~iscell(rec.cuts)
        error('night6:trimCut', ['the recovery-start record has no per-variable cuts ' ...
              '(RULING 2026-10-09 item 6)']);
    end
    if ~isfield(rec, 'analyses') || ~isstruct(rec.analyses)
        error('night6:trimOwner', 'the recovery-start record has no per-analysis starts');
    end
    for c = consumers(:)'
        if ~isfield(rec.analyses, c{1})
            error('night6:trimOwner', 'no recovery start recorded for run consumer %s', c{1});
        end
    end
    cuts = containers.Map();
    for k = 1:numel(rec.cuts)
        cuts(char(rec.cuts{k}.cut)) = rec.cuts{k};
    end
    kinds = OT.files;
    for f = files(:)'
        name = f{1};
        if any(strcmp(name, {'.', '..'})), continue, end
        hit = cellfun(@(k) ~isempty(regexp(name, k.pattern, 'once')), kinds);
        if nnz(hit) > 1
            error('night6:trimKind', '%s matches several output kinds', name);
        end
        if ~any(hit)
            T.untrimmed_files{end + 1} = struct('file', name, 'reason', ...
                'not in the output time map (a figure or an unknown file): left untrimmed');
            continue
        end
        t0 = tic;
        kind = kinds{hit}.kind;
        decl = OT.vars(cellfun(@(v) strcmp(v.file, kind), OT.vars));
        path = fullfile(outDir, name);
        S0 = load(path);
        if isfield(S0, OT.markerVariable)
            error('night6:trimTwice', '%s already holds %s: it was trimmed before', name, ...
                  OT.markerVariable);
        end
        tLoad = toc(t0);
        [S, info, marker] = trim_one(S0, decl, rec, cuts, consumers, fs, OT);
        clear S0
        S.(OT.markerVariable) = marker;
        tCut = toc(t0) - tLoad;
        save_like(path, S);
        clear S
        info.file = name;
        info.kind = kind;
        info.marker_variable = OT.markerVariable;
        info.wall_s = struct('load', tLoad, 'cut', tCut, 'save', toc(t0) - tLoad - tCut);
        T.files{end + 1} = info;
    end
end

% ==========================================================================
function [S, info, marker] = trim_one(S0, decl, rec, cuts, consumers, fs, OT)
    byPath = containers.Map();
    for k = 1:numel(decl), byPath(decl{k}.path) = decl{k}; end
    info = struct('trimmed', {{}}, 'epoch_scalars', {{}}, 'recomputed', {{}}, ...
                  'untrimmed_time_convention_unknown', {{}}, 'absent', {{}}, ...
                  'added_variables', {{}});
    info = classify(S0, '', byPath, info);
    i0 = double(rec.epoch_start_sample0);
    marker = struct('schema', 'gems-blanking-v2 night6 recovery trim v2', ...
        'ruling', [char(OT.ruling) ' (mode (B), per output variable); RULING 2026-10-08 (k) 2'], ...
        'mode', 'mask_to_electrical_drop_outputs', ...
        'meaning', ['An entry of a variable listed in vars whose stamp lies before ITS cut ' ...
                    '(cut: owner.output.class; trim_class valid_only = electrical settling ' ...
                    '+ its own input''s settling, filled_or_filtered = electrical settling ' ...
                    '+ its full reach) was NOT COMPUTED. action nan: set NaN; action ' ...
                    'nan_events: the event series is double (1 event, 0 no event) and NaN ' ...
                    'there (never "no event"); action drop: removed (n_dropped). ' ...
                    'not_computed_rows are 1-based inclusive [first last] runs of rows (a ' ...
                    'matrix) or elements (a list) of the variable as saved. ' ...
                    'first_computed_epoch_row is the cut as a 1-based epoch row; ' ...
                    'first_computed_epoch_sample0 the same as a 0-based epoch sample; ' ...
                    'start_sample0 a 0-based file sample. valid_fraction names the variable ' ...
                    'holding the fraction of valid input rows in each value''s window ' ...
                    '(added_variables lists the ones Night 6 added). recomputed holds every ' ...
                    'whole-epoch average or count recomputed from the kept values, with ' ...
                    'her original. epoch_scalars were computed over [electrical settling, ' ...
                    'epoch end]; untrimmed variables have no known time convention and were ' ...
                    'not cut.'], ...
        'fs', fs, 'epoch_start_sample0', i0, ...
        'electrical_settle_sample0', field_or_empty(rec, 'electrical_settle_sample0'), ...
        'run_consumers', {consumers(:)'}, 'trim_classes', OT.trimClasses, 'vars', {{}}, ...
        'added_variables', {{}}, 'recomputed', {{}}, ...
        'epoch_scalars', {info.epoch_scalars}, ...
        'untrimmed', {cellfun(@(e) e.path, info.untrimmed_time_convention_unknown, ...
                              'UniformOutput', false)});
    S = S0;
    for k = 1:numel(decl)
        d = decl{k};
        if ~strcmp(d.role, 'trim'), continue, end
        if ~isfield(d, 'trim_class') || ~ischar(d.trim_class) ...
                || ~isfield(OT.trimClasses, d.trim_class)
            error('night6:trimClass', ['%s has no known trim class (RULING 2026-10-09 ' ...
                  'item 6: valid_only or filled_or_filtered, declared per variable; never ' ...
                  'guessed)'], d.path);
        end
        vparts = strsplit(d.path, '.');
        if ~has_path(S0, vparts)
            info.absent{end + 1} = d.path;
            continue
        end
        if ~isfield(d, 'cut') || ~isKey(cuts, d.cut)
            error('night6:trimCut', ['%s is cut at %s, which this file''s row does not ' ...
                  'hold: it is never cut at another variable''s point'], d.path, ...
                  field_or_text(d, 'cut'));
        end
        C = cuts(d.cut);
        Lo = double(C.output_rows_before_start);
        owner = char(d.owner);
        if any(strcmp(consumers, owner))
            basis = sprintf('cut %s', d.cut);
        else
            basis = sprintf('cut %s (a byproduct of a [%s] run: %s''s own cut)', d.cut, ...
                            strjoin(consumers, ','), owner);
        end
        sparts = strsplit(d.stamp, '.');
        sibling = numel(vparts) == numel(sparts) && isequal(vparts(1:end - 1), sparts(1:end - 1));
        fixed = [];
        if ~sibling, fixed = resolve_scalar(S0, sparts, d.path); end
        conv = struct('name', d.convention, 'fs', fs, 'W', [], 'S', []);
        if strcmp(d.convention, 'sec_xchan_delay')
            conv.W = double(resolve_scalar(S0, strsplit(OT.xchanDelayParams{1}, '.'), d.path));
            conv.S = double(resolve_scalar(S0, strsplit(OT.xchanDelayParams{2}, '.'), d.path));
        end
        [S, n, leaves] = apply(S, S0, vparts, sparts, sibling, fixed, conv, d.action, Lo, ...
                               d.path, '');
        info.trimmed{end + 1} = struct('path', d.path, 'owner', owner, 'start_basis', basis, ...
            'cut', d.cut, 'trim_class', d.trim_class, 'action', d.action, ...
            'convention', d.convention, 'stamp', d.stamp, 'rows_before_start', Lo, ...
            'n_entries', n);
        meaning = '';
        if isfield(OT.conventions, d.convention), meaning = OT.conventions.(d.convention); end
        edge = struct('rule', '', 'source', '', 'half_valid', false);
        if isfield(d, 'edge') && isstruct(d.edge), edge = d.edge; end
        vfr = [];
        if isfield(d, 'valid_fraction') && isstruct(d.valid_fraction)
            [S, vfr] = add_valid_fraction(S, S0, d, sparts, sibling, fixed, fs, OT.suffix);
            if vfr.added
                info.added_variables{end + 1} = vfr.variable;
                marker.added_variables{end + 1} = vfr.variable;
            end
        end
        marker.vars{end + 1} = struct('path', d.path, 'owner', owner, 'start_basis', basis, ...
            'trim_class', d.trim_class, 'trim_class_meaning', OT.trimClasses.(d.trim_class), ...
            'cut', d.cut, 'output_key', d.output_key, 'cut_basis', char(C.basis), ...
            'edge_rule', edge, 'action', d.action, 'stamp', d.stamp, ...
            'convention', d.convention, 'convention_meaning', meaning, 'mode', marker.mode, ...
            'start_s', double(C.start_s), 'start_sample0', double(C.start_sample0), ...
            'first_computed_epoch_row', Lo + 1, 'first_computed_epoch_sample0', Lo, ...
            'n_not_computed', n, 'leaves', {leaves}, 'valid_fraction', vfr);
    end
    for k = 1:numel(decl)
        d = decl{k};
        if ~strcmp(d.role, 'recomputed'), continue, end
        if ~has_path(S0, strsplit(d.path, '.')) || ~has_path(S0, strsplit(d.recompute.of, '.'))
            info.absent{end + 1} = d.path;
            continue
        end
        [S, e] = recompute(S, S0, d, fs);
        marker.recomputed{end + 1} = e;
        info.recomputed{end + 1} = d.path;
    end
end

function v = field_or_empty(s, f)
    v = [];
    if isfield(s, f), v = double(s.(f)); end
end

function t = field_or_text(s, f)
    t = '(none)';
    if isfield(s, f), t = char(s.(f)); end
end

function info = classify(v, prefix, byPath, info)
% Walk every field (struct arrays by their fields); list epoch scalars and the untrimmed.
    if ~isstruct(v), return, end
    for fn = fieldnames(v)'
        p = fn{1};
        if ~isempty(prefix), p = [prefix '.' fn{1}]; end
        if ~isKey(byPath, p)
            info.untrimmed_time_convention_unknown{end + 1} = struct('path', p, 'why', ...
                'not in the output time map: left untrimmed');
            continue
        end
        d = byPath(p);
        switch d.role
            case 'container'
                if ~isempty(v)
                    info = classify([v.(fn{1})], p, byPath, info);
                end
            case 'epoch_scalar'
                info.epoch_scalars{end + 1} = p;
            case 'unknown'
                w = 'time convention unknown: left untrimmed';
                if isfield(d, 'why'), w = sprintf('%s (%s)', w, d.why); end
                info.untrimmed_time_convention_unknown{end + 1} = struct('path', p, 'why', w);
        end
    end
end

function tf = has_path(S, parts)
    tf = true;
    v = S;
    for k = 1:numel(parts)
        if ~isstruct(v) || ~isfield(v, parts{k}), tf = false; return, end
        if isempty(v), return, end
        v = v(1).(parts{k});
    end
end

function v = resolve_scalar(S, parts, what)
% An absolute stamp (or parameter) path: every struct on the way must be scalar.
    v = S;
    for k = 1:numel(parts)
        if ~isstruct(v) || ~isscalar(v) || ~isfield(v, parts{k})
            error('night6:trimShape', '%s: cannot resolve %s as one value', what, ...
                  strjoin(parts, '.'));
        end
        v = v.(parts{k});
    end
end

function [Snew, n, leaves] = apply(Snew, S0, vparts, sparts, sibling, fixed, conv, action, ...
                                   Lo, what, label)
% Descend both paths together through (possibly arrayed) structs; cut at the leaf.
    n = 0;
    leaves = {};
    if numel(vparts) == 1
        stamp = fixed;
        if sibling, stamp = S0.(sparts{1}); end
        [Snew.(vparts{1}), n, leaves] = cut(S0.(vparts{1}), stamp, conv, action, Lo, what, ...
                                             [label vparts{1}]);
        return
    end
    head = vparts{1};
    sub0 = S0.(head);
    subNew = Snew.(head);
    for i = 1:numel(sub0)
        sp = sparts;
        if sibling, sp = sparts(2:end); end
        sub = sprintf('%s%s.', label, head);
        if numel(sub0) > 1, sub = sprintf('%s%s(%d).', label, head, i); end
        [e, m, lv] = apply(subNew(i), sub0(i), vparts(2:end), sp, sibling, fixed, conv, ...
                           action, Lo, what, sub);
        subNew(i) = e;
        n = n + m;
        leaves = [leaves, lv]; %#ok<AGROW>
    end
    Snew.(head) = subNew;
end

function [val, n, leaves] = cut(val, stamp, conv, action, Lo, what, label)
    if iscell(val)
        if ~iscell(stamp) || numel(stamp) ~= numel(val)
            error('night6:trimShape', '%s: a cell value needs a cell stamp of the same size', what);
        end
        n = 0;
        leaves = {};
        for k = 1:numel(val)
            [val{k}, m, lv] = cut(val{k}, stamp{k}, conv, action, Lo, what, ...
                                  sprintf('%s{%d}', label, k));
            n = n + m;
            leaves = [leaves, lv]; %#ok<AGROW>
        end
        return
    end
    if ~isnumeric(stamp)
        error('night6:trimShape', '%s: its stamp is not numeric', what);
    end
    before = position(double(stamp(:)), conv) < Lo - 1e-6;
    m = numel(before);
    n = nnz(before);
    leaf = struct('leaf', label, 'n_entries', m, 'n_not_computed', n, ...
                  'not_computed_rows', zeros(0, 2), 'n_dropped', 0, 'class_before', class(val));
    if isempty(val) && m == 0
        leaves = {leaf};
        return
    end
    rows = list_or_rows(val, m, what);
    switch action
        case 'drop'
            if rows, val(before, :) = []; else, val(before) = []; end
            leaf.n_dropped = n;
        case 'nan'
            if ~isfloat(val)
                error('night6:trimShape', '%s: action nan needs a floating value, got %s', ...
                      what, class(val));
            end
            if rows, val(before, :) = NaN; else, val(before) = NaN; end
            leaf.not_computed_rows = runs(before);
        case 'nan_events'
            if ~islogical(val)
                error('night6:trimShape', '%s: action nan_events needs her logical event series, got %s', ...
                      what, class(val));
            end
            val = double(val);                % 1 event, 0 no event, NaN not computed
            if rows, val(before, :) = NaN; else, val(before) = NaN; end
            leaf.not_computed_rows = runs(before);
        otherwise
            error('night6:trimShape', '%s: unknown action %s', what, action);
    end
    leaves = {leaf};
end

function rows = list_or_rows(val, m, what)
    if isvector(val) && numel(val) == m && ~(size(val, 1) == m && size(val, 2) > 1)
        rows = false;                         % a list: index its elements
    elseif size(val, 1) == m
        rows = true;                          % a matrix: one row per stamp
    else
        error('night6:trimShape', '%s: %d stamps for a value of size %s', what, m, ...
              mat2str(size(val)));
    end
end

function r = runs(b)
% 1-based inclusive [first last] runs of true, one per row.
    e = diff([false; b(:); false]);
    r = [find(e == 1), find(e == -1) - 1];
end

function p = position(v, conv)
% The map's conventions (extent.recovery_start.CONVENTIONS): 0-based epoch samples.
    switch conv.name
        case 'row1',            p = v - 1;
        case 'sec0',            p = v * conv.fs;
        case 'sec_row1',        p = v * conv.fs - 1;
        case 'sec_xchan_delay', p = (v + conv.W / 2 - conv.S) * conv.fs;
        otherwise
            error('night6:trimShape', 'unknown stamp convention %s', conv.name);
    end
end

% ==========================================================================
% valid fractions (RULING 2026-10-09 item 6)

function [S, vfr] = add_valid_fraction(S, S0, d, sparts, sibling, fixed, fs, suffix)
% The fraction of valid input rows in each value's window, over every leaf of d.path.
    vf = d.valid_fraction;
    vfr = struct('kind', vf.kind, 'source', vf.source, 'variable', '', 'added', false);
    if strcmp(vf.kind, 'her')
        vfr.variable = vf.variable;           % her own sibling already holds it
        return
    end
    vparts = strsplit(d.path, '.');
    L = leaf_chains(S0, vparts);
    for i = 1:numel(L)
        ch = L{i}.chain;
        val = subsref(S0, ch);
        if sibling
            stamp = subsref(S0, [ch(1:end - 1), substruct('.', sparts{end})]);
        else
            stamp = fixed;
        end
        parent = S0;
        if numel(ch) > 1, parent = subsref(S0, ch(1:end - 1)); end
        sib = [vparts{end} suffix];
        if isfield(parent, sib)
            error('night6:trimShape', '%s: its sibling %s already exists', d.path, sib);
        end
        if iscell(val)
            fr = cell(size(val));
            for c = 1:numel(val)
                fr{c} = fraction(vf, val{c}, stamp{c}, c, S0, fs, d.path);
            end
        else
            fr = fraction(vf, val, stamp, L{i}.k, S0, fs, d.path);
        end
        S = subsasgn(S, [ch(1:end - 1), substruct('.', sib)], fr);
    end
    vfr.variable = [d.path suffix];
    vfr.added = true;
end

function L = leaf_chains(S, parts)
% Every occurrence of the leaf: its subsref chain and k, the index of the struct-array
% element on the way (1 when every struct on the way is scalar).
    L = {struct('chain', struct('type', {}, 'subs', {}), 'k', 1)};
    for p = 1:numel(parts) - 1
        nxt = {};
        for i = 1:numel(L)
            v = subsref(S, [L{i}.chain, substruct('.', parts{p})]);
            for e = 1:numel(v)
                k = L{i}.k;
                if numel(v) > 1, k = e; end
                nxt{end + 1} = struct('chain', [L{i}.chain, substruct('.', parts{p}, '()', {e})], ...
                                      'k', k); %#ok<AGROW>
            end
        end
        L = nxt;
    end
    for i = 1:numel(L), L{i}.chain = [L{i}.chain, substruct('.', parts{end})]; end
end

function fr = fraction(vf, val, stamp, k, S0, fs, what)
% One leaf's fractions, the shape of its value (VALID_FRACTION_KINDS, Python).
    switch vf.kind
        case 'hr_window'        % HR_BR_HRVAnalysis_beats.m:835-838, :873-876
            W = width(vf, S0, what);
            valid = ~logical(resolve_scalar(S0, strsplit(vf.validity, '.'), what));
            N = numel(valid);
            tc = double(stamp(:));
            lo = max(1, round((tc - W / 2) * fs) + 1);
            hi = min(N, round((tc + W / 2) * fs) + 1);
            fr = shape(window_frac(valid(:), lo, hi), val, what);
        case 'sw_window'        % slowWaveAnalysis_new.m:229, :246-247
            W = width(vf, S0, what);
            valid = ~logical(resolve_scalar(S0, strsplit(vf.validity, '.'), what));
            N = numel(valid);
            c = round(double(stamp(:)) * fs) + 1;
            h = round(W * fs / 2);
            fr = shape(window_frac(valid(:), max(1, c - h), min(N, c + h)), val, what);
        case 'mmc_rate_window'  % extract_mmc.m:284-290
            W = width(vf, S0, what);
            sig = resolve_scalar(S0, strsplit(vf.validity, '.'), what);
            N = size(sig, 1);
            c = double(stamp(:));
            lo = max(1, floor((c - W / 2) * fs) + 1);
            hi = min(N, floor((c + W / 2) * fs));
            if size(val, 1) ~= numel(c) || size(val, 2) ~= size(sig, 2)
                error('night6:trimShape', '%s: %s rate rows for %d windows', what, ...
                      mat2str(size(val)), numel(c));
            end
            fr = zeros(size(val));
            for ch = 1:size(val, 2)
                fr(:, ch) = window_frac(~isnan(sig(:, ch)), lo, hi);
            end
        case 'mmc_delay_window' % extract_mmc.m:259-268 (xchan_delay on the firing rate)
            P = resolve_scalar(S0, {'mmc', 'params'}, what);
            rate = resolve_scalar(S0, strsplit(vf.validity, '.'), what);
            M = size(rate, 1);
            wlen = max(4, round(double(P.delayW) / double(P.S)));
            step = max(1, round(double(P.delayStep) / double(P.S)));
            starts = 1:step:max(1, M - wlen + 1);
            pairs = [1 2; 1 3; 2 3];
            if numel(starts) ~= size(val, 1) || size(val, 2) ~= size(pairs, 1)
                error('night6:trimShape', '%s: %d delay windows for a value of size %s', ...
                      what, numel(starts), mat2str(size(val)));
            end
            fr = zeros(size(val));
            for s = 1:numel(starts)
                lo = starts(s);
                hi = min(M, lo + wlen - 1);
                for p = 1:size(pairs, 1)
                    ok = isfinite(rate(lo:hi, pairs(p, 1))) & isfinite(rate(lo:hi, pairs(p, 2)));
                    fr(s, p) = nnz(ok) / (hi - lo + 1);
                end
            end
        case 'step2_window'     % step2_noise_sigma.m:42-43, :90-91
            N = double(resolve_scalar(S0, {'nSamples'}, what));
            valid = valid_from_runs(S0, vf.validity, k, N, what);
            win = max(2, round(width(vf, S0, what) * fs));
            step = max(1, round(win * double(resolve_scalar(S0, {'sigmaWin', 'stepFrac'}, what))));
            nWin = max(1, floor((N - win) / step) + 1);
            if numel(val) ~= nWin
                error('night6:trimShape', '%s: %d sigma windows for %d values', what, nWin, ...
                      numel(val));
            end
            lo = ((1:nWin)' - 1) * step + 1;
            fr = shape(window_frac(valid, lo, min(N, lo + win - 1)), val, what);
        case 'cv2_bins'         % step6_spike_report.m:263, :267 (rolling_cv2)
            N = double(resolve_scalar(S0, {'nSamples'}, what));
            valid = valid_from_runs(S0, vf.validity, k, N, what);
            winSec = width(vf, S0, what);
            Tend = (N - 1) / fs;
            edges = 0:winSec:Tend;
            if edges(end) < Tend, edges(end + 1) = Tend; end
            nb = numel(edges) - 1;
            if numel(val) ~= nb
                error('night6:trimShape', '%s: %d CV2 bins for %d values', what, nb, numel(val));
            end
            lo = arrayfun(@(e) first_row_at_or_after(e, fs, N), edges(1:end - 1)');
            hi = arrayfun(@(e) first_row_at_or_after(e, fs, N), edges(2:end)') - 1;
            fr = shape(window_frac(valid, lo, hi), val, what);
        otherwise
            error('night6:trimShape', '%s: unknown valid-fraction kind %s', what, vf.kind);
    end
end

function W = width(vf, S0, what)
    if isfield(vf, 'width') && ~isempty(vf.width)
        W = double(resolve_scalar(S0, strsplit(vf.width, '.'), what));
    else
        W = double(vf.width_s);
    end
end

function f = window_frac(valid, lo, hi)
% Fraction of valid rows in each window [lo, hi] (1-based inclusive); NaN for an empty one.
    cv = [0; cumsum(double(valid(:)))];
    lo = double(lo(:));
    hi = double(hi(:));
    f = (cv(max(hi, lo - 1) + 1) - cv(lo)) ./ (hi - lo + 1);
    f(hi < lo) = NaN;
end

function fr = shape(f, val, what)
    if size(val, 1) == numel(f)
        fr = repmat(f, 1, size(val, 2));      % one row per window (columns share it)
    elseif numel(val) == numel(f)
        fr = reshape(f, size(val));
    else
        error('night6:trimShape', '%s: %d windows for a value of size %s', what, numel(f), ...
              mat2str(size(val)));
    end
end

function v = valid_from_runs(S0, name, k, N, what)
% The channel's validMask, from her invalid runs as save_spikes_v2 stored them.
    R = resolve_scalar(S0, {name}, what);
    if ~iscell(R) || k > numel(R)
        error('night6:trimShape', '%s: no %s for channel %d', what, name, k);
    end
    r = double(R{k});
    v = true(N, 1);
    for j = 1:size(r, 1)
        v(r(j, 1):r(j, 2)) = false;
    end
end

function i = first_row_at_or_after(e, fs, N)
% The first 1-based row whose time (row - 1) / fs is >= e - her own comparison.
    i = max(1, ceil(e * fs) + 1);
    while i > 1 && (i - 2) / fs >= e, i = i - 1; end
    while i <= N && (i - 1) / fs < e, i = i + 1; end
end

% ==========================================================================
% recomputed averages (RULING 2026-10-09 item 6)

function [S, e] = recompute(S, S0, d, fs)
    rc = d.recompute;
    L = leaf_chains(S0, strsplit(d.path, '.'));
    leaves = cell(1, numel(L));
    for i = 1:numel(L)
        ch = L{i}.chain;
        orig = subsref(S0, ch);
        x = by_element(S, rc.of, L{i}.k, numel(L), d.path);
        switch rc.kind
            case 'mean_omitnan'
                new = mean(x, 1, 'omitnan');
            case 'count'
                new = numel(x);
            case 'mean_well_sampled'
                w = by_element(S, rc.where, L{i}.k, numel(L), d.path);
                fin = by_element(S, rc.finite, L{i}.k, numel(L), d.path);
                ws = w >= 0.5 & isfinite(fin);
                new = mean(x(ws), 'omitnan');
            case 'events_per_valid_s'
                sig = by_element(S0, rc.validity, L{i}.k, numel(L), d.path);
                kept = ~isnan(x);
                new = zeros(1, size(x, 2));
                for c = 1:size(x, 2)
                    new(c) = nnz(x(kept(:, c), c) == 1) / ...
                             max(nnz(kept(:, c) & ~isnan(sig(:, c))) / fs, eps);
                end
            otherwise
                error('night6:trimShape', '%s: unknown recompute kind %s', d.path, rc.kind);
        end
        if ~isequal(size(new), size(orig))
            error('night6:trimShape', '%s: recomputed %s for her %s', d.path, ...
                  mat2str(size(new)), mat2str(size(orig)));
        end
        new = cast(new, class(orig));
        S = subsasgn(S, ch, new);
        leaves{i} = struct('leaf', chain_label(ch), 'original', orig, 'recomputed', new);
    end
    e = struct('path', d.path, 'kind', rc.kind, 'of', rc.of, 'source', d.source, ...
               'label', ['RECOMPUTED from the kept values (RULING 2026-10-09 item 6); ' ...
                         'original is her value as the call computed it'], ...
               'leaves', {leaves});
end

function v = by_element(S, path, k, n, what)
% A path read at struct-array element k (the target's element), scalar structs as they are.
    v = S;
    for p = strsplit(path, '.')
        if ~isstruct(v) || ~isfield(v, p{1})
            error('night6:trimShape', '%s: cannot read %s', what, path);
        end
        if numel(v) > 1
            if numel(v) ~= n
                error('night6:trimShape', '%s: %s has %d elements for %d', what, path, ...
                      numel(v), n);
            end
            v = v(k);
        end
        v = v.(p{1});
    end
end

function s = chain_label(ch)
    s = '';
    for c = ch
        if strcmp(c.type, '.')
            if isempty(s), s = c.subs; else, s = [s '.' c.subs]; end %#ok<AGROW>
        else
            s = sprintf('%s(%d)', s, c.subs{1});
        end
    end
end

function save_like(path, S)
% Rewrite atomically in the file's own MAT version (v7.3 = HDF5, else the default).
    fid = fopen(path, 'r');
    head = fread(fid, [1 128], '*char');
    fclose(fid);
    tmp = [path(1:end - 4) '.trimtmp.mat'];
    if contains(head, 'MATLAB 7.3')
        save(tmp, '-struct', 'S', '-v7.3');
    else
        save(tmp, '-struct', 'S');
    end
    movefile(tmp, path, 'f');
end
