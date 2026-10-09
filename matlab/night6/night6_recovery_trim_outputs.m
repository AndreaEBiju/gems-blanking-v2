function T = night6_recovery_trim_outputs(outDir, files, consumers, rec, OT, fs)
% NIGHT6_RECOVERY_TRIM_OUTPUTS  Drop or flag every output stamped before its analysis's start.
%
%   T = night6_recovery_trim_outputs(outDir, files, consumers, rec, OT, fs)
%
%   outDir     the epoch's output folder
%   files      the names of the files this run created (after night6_keep_slow_wave)
%   consumers  the run's consumers
%   rec        plan.recoveryStart (mode mask_to_electrical_drop_outputs, applies true)
%   OT         RS.outputTimes: the output time map (Python extent.recovery_start.OUTPUT_VARS)
%   fs         the epoch's sample rate
%
% RULING 2026-10-08 (k) 2, mode mask_to_electrical_drop_outputs. The input of every
% analysis was masked before the electrical settling; here each output file the run kept
% is matched to its kind in the map, loaded, and every variable classified there as
% 'trim' is cut at its OWNER's start:
%
%   L    = rec.consumers.<owner>.output_rows_before_start (epoch rows before the start);
%          an owner that is not one of this run's consumers (a byproduct, e.g. the breath
%          rate of an hrv-only run) takes the latest start among the run's consumers;
%   p    = the stamp's 0-based epoch position by its convention (row1, sec0, sec_row1,
%          sec_xchan_delay - the map's definitions), in samples;
%   an entry is BEFORE the start iff p < L - 1e-6 (a stamp of an integer row computed in
%          floating point counts as that row, never as the one before it);
%   action nan   the value is set NaN (her own not-computed marker), the stamp is kept;
%          drop  the entry is removed from its list, with every list on the same stamps;
%          false a logical event series: rows set false, named in the record.
%
% Every stamp is read from the file as the call wrote it, before anything is changed, so
% co-indexed lists are cut by the same entries. Everything else is untouched: a value at
% or after the start is bit-identical to the call's own output. The file is rewritten
% atomically in its own MAT version.
%
% Never silent: an epoch-wide scalar is listed (computed over [electrical settling, epoch
% end]); a variable the map marks 'unknown', or that the map does not list, is left
% UNTRIMMED and listed by name; a file matching no kind (a figure) is listed untrimmed; a
% stamp whose length does not match its value is refused ('night6:trimShape').
    T = struct('mode', 'mask_to_electrical_drop_outputs', 'files', {{}}, ...
               'untrimmed_files', {{}});
    L = struct();
    for c = fieldnames(rec.consumers)'
        L.(c{1}) = double(rec.consumers.(c{1}).output_rows_before_start);
    end
    runL = 0;
    for c = consumers(:)'
        if ~isfield(L, c{1})
            error('night6:trimOwner', 'no recovery start recorded for run consumer %s', c{1});
        end
        runL = max(runL, L.(c{1}));
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
        kind = kinds{hit}.kind;
        decl = OT.vars(cellfun(@(v) strcmp(v.file, kind), OT.vars));
        path = fullfile(outDir, name);
        S0 = load(path);
        [S, info] = trim_one(S0, decl, L, consumers, runL, fs, OT);
        info.file = name;
        info.kind = kind;
        save_like(path, S);
        T.files{end + 1} = info;
    end
end

% ==========================================================================
function [S, info] = trim_one(S0, decl, L, consumers, runL, fs, OT)
    byPath = containers.Map();
    for k = 1:numel(decl), byPath(decl{k}.path) = decl{k}; end
    info = struct('trimmed', {{}}, 'epoch_scalars', {{}}, ...
                  'untrimmed_time_convention_unknown', {{}}, 'absent', {{}});
    info = classify(S0, '', byPath, info);
    S = S0;
    for k = 1:numel(decl)
        d = decl{k};
        if ~strcmp(d.role, 'trim'), continue, end
        vparts = strsplit(d.path, '.');
        if ~has_path(S0, vparts)
            info.absent{end + 1} = d.path;
            continue
        end
        owner = char(d.owner);
        if any(strcmp(consumers, owner))
            Lo = L.(owner); basis = owner;
        else
            Lo = runL; basis = sprintf('byproduct of a [%s] run: the run''s latest start', ...
                                       strjoin(consumers, ','));
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
        [S, n] = apply(S, S0, vparts, sparts, sibling, fixed, conv, d.action, Lo, d.path);
        info.trimmed{end + 1} = struct('path', d.path, 'owner', owner, 'start_basis', basis, ...
            'action', d.action, 'convention', d.convention, 'stamp', d.stamp, ...
            'rows_before_start', Lo, 'n_entries', n);
    end
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

function [Snew, n] = apply(Snew, S0, vparts, sparts, sibling, fixed, conv, action, Lo, what)
% Descend both paths together through (possibly arrayed) structs; cut at the leaf.
    n = 0;
    if numel(vparts) == 1
        stamp = fixed;
        if sibling, stamp = S0.(sparts{1}); end
        [Snew.(vparts{1}), n] = cut(S0.(vparts{1}), stamp, conv, action, Lo, what);
        return
    end
    head = vparts{1};
    sub0 = S0.(head);
    subNew = Snew.(head);
    for i = 1:numel(sub0)
        sp = sparts;
        if sibling, sp = sparts(2:end); end
        [e, m] = apply(subNew(i), sub0(i), vparts(2:end), sp, sibling, fixed, conv, action, ...
                       Lo, what);
        subNew(i) = e;
        n = n + m;
    end
    Snew.(head) = subNew;
end

function [val, n] = cut(val, stamp, conv, action, Lo, what)
    if iscell(val)
        if ~iscell(stamp) || numel(stamp) ~= numel(val)
            error('night6:trimShape', '%s: a cell value needs a cell stamp of the same size', what);
        end
        n = 0;
        for k = 1:numel(val)
            [val{k}, m] = cut(val{k}, stamp{k}, conv, action, Lo, what);
            n = n + m;
        end
        return
    end
    if ~isnumeric(stamp)
        error('night6:trimShape', '%s: its stamp is not numeric', what);
    end
    before = position(double(stamp(:)), conv) < Lo - 1e-6;
    m = numel(before);
    n = nnz(before);
    if isempty(val) && m == 0, return, end
    if isvector(val) && numel(val) == m && ~(size(val, 1) == m && size(val, 2) > 1)
        rows = false;                         % a list: index its elements
    elseif size(val, 1) == m
        rows = true;                          % a matrix: one row per stamp
    else
        error('night6:trimShape', '%s: %d stamps for a value of size %s', what, m, ...
              mat2str(size(val)));
    end
    switch action
        case 'drop'
            if rows, val(before, :) = []; else, val(before) = []; end
        case 'nan'
            if ~isfloat(val)
                error('night6:trimShape', '%s: action nan needs a floating value, got %s', ...
                      what, class(val));
            end
            if rows, val(before, :) = NaN; else, val(before) = NaN; end
        case 'false'
            if ~islogical(val)
                error('night6:trimShape', '%s: action false needs a logical, got %s', what, ...
                      class(val));
            end
            if rows, val(before, :) = false; else, val(before) = false; end
        otherwise
            error('night6:trimShape', '%s: unknown action %s', what, action);
    end
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
