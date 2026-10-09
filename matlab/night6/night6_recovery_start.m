function RS = night6_recovery_start(file)
% NIGHT6_RECOVERY_START  Read the declared recovery-starts file (RULING 2026-10-08 (k) 2).
%
%   RS = night6_recovery_start(file)
%
% The file is written by gems_blanking_v2.extent.recovery_start.write_recovery_starts
% (schema 'gems-blanking-v2 recovery starts v1'): per stim_rec file and per analysis
% (= consumer), the analysis's start as an exact 0-based FILE sample (start_sample0) with
% its basis and source, and the files held for want of stim edges or settling. It is a
% DECLARED input: its path and SHA-256 go into every record (RS.file, RS.sha256).
% An empty path returns [] - Night 6 then refuses every stim_recovery epoch by name
% (night6_recovery_lead_in), never runs one untrimmed.
%
% Refused here by name ('night6:recoveryStartFile'): an unreadable file, another schema,
% a session listed twice (case-insensitively, as the store matches names), a file both
% measured and held, an analysis listed twice for one file, and a start_sample0 that is
% not one integer >= 0. Rows are matched by strcmp on the session and analysis STRINGS,
% never by jsondecode field names (which mangle keys - invariant 22).
    if isempty(file)
        RS = [];
        return
    end
    file = char(file);
    if ~isfile(file)
        error('night6:recoveryStartFile', 'the recovery-starts file %s does not exist', file);
    end
    D = jsondecode(fileread(file));
    want = 'gems-blanking-v2 recovery starts v1';
    if ~isstruct(D) || ~isfield(D, 'schema') || ~strcmp(D.schema, want)
        error('night6:recoveryStartFile', '%s is not a ''%s'' file', file, want);
    end
    for f = {'fs', 'fixed_start_s', 'files', 'held'}
        if ~isfield(D, f{1})
            error('night6:recoveryStartFile', '%s has no %s', file, f{1});
        end
    end
    RS = struct('file', file, 'sha256', night6_sha256_file(file), 'schema', D.schema, ...
                'fs', double(D.fs), 'fixed_start_s', double(D.fixed_start_s));
    RS.files = as_cells(D.files);
    RS.held = as_cells(D.held);
    seen = {};
    for k = 1:numel(RS.files)
        F = RS.files{k};
        seen = add_session(seen, F, file);
        rows = as_cells(F.analyses);
        names = cellfun(@(r) char(r.analysis), rows, 'UniformOutput', false);
        if numel(unique(names)) ~= numel(names)
            error('night6:recoveryStartFile', '%s: %s lists an analysis twice', file, F.session);
        end
        for j = 1:numel(rows)
            k0 = rows{j}.start_sample0;
            if ~isnumeric(k0) || ~isscalar(k0) || k0 ~= fix(k0) || k0 < 0
                error('night6:recoveryStartFile', ['%s: %s/%s start_sample0 must be one ' ...
                      'integer >= 0'], file, F.session, names{j});
            end
        end
        RS.files{k}.analyses = rows;
    end
    for k = 1:numel(RS.held)
        seen = add_session(seen, RS.held{k}, file);
    end
end

function c = as_cells(v)
% jsondecode gives a struct array when every element has the same fields, a cell array
% when they differ, and [] for an empty array.
    if isempty(v)
        c = {};
    elseif isstruct(v)
        c = num2cell(v(:))';
    else
        c = v(:)';
    end
end

function seen = add_session(seen, F, file)
    if ~isfield(F, 'session') || ~ischar(F.session)
        error('night6:recoveryStartFile', '%s: a row has no session', file);
    end
    if any(strcmpi(seen, F.session))
        error('night6:recoveryStartFile', '%s: session %s appears twice', file, F.session);
    end
    seen{end + 1} = F.session;
end
