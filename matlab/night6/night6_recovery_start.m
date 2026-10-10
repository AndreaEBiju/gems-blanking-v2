function RS = night6_recovery_start(file)
% NIGHT6_RECOVERY_START  Read the declared recovery-starts file (RULING 2026-10-08 (k) 2).
%
%   RS = night6_recovery_start(file)
%
% The file is written by gems_blanking_v2.extent.recovery_start.write_recovery_starts
% (schema 'gems-blanking-v2 recovery starts v5'): per stim_rec file BOTH stim times
% (RULING 2026-10-09 (i) 3 (c): electrical_end_s, the stimulator's AmA record, and
% mechanical_end_s, the gate/MotorOn offset), the electrical settling and the mechanical
% end as exact 0-based FILE samples (electrical_settle_sample0, mechanical_end_sample0),
% and the INPUT MASK point input_mask_sample0 = their maximum, per analysis
% (= consumer) the analysis's start as one (start_sample0) with its basis and source, and
% per CUT (owner.output.class, RULING 2026-10-09 item 6) the cut point as one; the output
% time map (RS.outputTimes: per-variable stamps, classes, cuts, valid fractions and
% recomputed averages - Python's extent.recovery_start.OUTPUT_VARS); the edge settlings
% the cuts were derived under (edge_settling, edge_settling_sha256); and the files held
% for want of stim edges or settling. A v1 or v2 file (no cuts: the owner-start cut is
% not mode (B)), a v3 file (no edge-settling hash) and a v4 file (one stim time) are
% refused by name. It is a
% DECLARED input: its path and SHA-256 go into every record (RS.file, RS.sha256).
% An empty path returns [] - Night 6 then refuses every stim_recovery epoch by name
% (night6_recovery_lead_in), never runs one untrimmed.
%
% Refused here by name ('night6:recoveryStartFile'): an unreadable file, another schema,
% a session listed twice (case-insensitively, as the store matches names), a file both
% measured and held, an analysis or a cut listed twice for one file, a file with no
% cuts, a start_sample0, electrical_settle_sample0, mechanical_end_sample0 or
% input_mask_sample0 that is not one integer >= 0, a file without both stim times or a
% positive fs, an input_mask_sample0 that is not max(electrical_settle_sample0,
% mechanical_end_sample0), any of those three samples that is not the first sample at or
% after its seconds field (night6_first_sample), an electrical_settle_s before the
% electrical_end_s, a measured start or cut before the input mask point, and
% a missing output map or source list. Refused as 'night6:recoveryStartSource', naming
% the file: a cited processing_new file (source_files, Python SOURCE_FILES) that is not on the path, or
% whose SHA-256 is not the one the starts file was computed from. Refused as
% 'night6:recoveryStartEdgeSettling': a file whose edge_settling_sha256 is not
% night6_edge_settling()'s sha256, or whose edge_settling differs from that declaration -
% starts derived under other edge settlings are stale (review 2026-10-09). Rows are matched by
% strcmp on the session and analysis STRINGS, never by jsondecode field names (which
% mangle keys - invariant 22).
    if isempty(file)
        RS = [];
        return
    end
    file = char(file);
    if ~isfile(file)
        error('night6:recoveryStartFile', 'the recovery-starts file %s does not exist', file);
    end
    D = jsondecode(fileread(file));
    want = 'gems-blanking-v2 recovery starts v5';
    if ~isstruct(D) || ~isfield(D, 'schema') || ~strcmp(D.schema, want)
        error('night6:recoveryStartFile', '%s is not a ''%s'' file', file, want);
    end
    for f = {'fs', 'fixed_start_s', 'files', 'held', 'output_times', 'source_files', ...
             'edge_settling', 'edge_settling_sha256'}
        if ~isfield(D, f{1})
            error('night6:recoveryStartFile', '%s has no %s', file, f{1});
        end
    end
    check_sources(D.source_files, file);   % the code the starts were read from IS the code run
    Ed = check_edge_settling(D, file);     % ... and under the edge settlings applied now
    RS = struct('file', file, 'sha256', night6_sha256_file(file), 'schema', D.schema, ...
                'fs', double(D.fs), 'fixed_start_s', double(D.fixed_start_s), ...
                'edgeSettlingSha256', Ed.sha256);
    RS.outputTimes = output_times(D.output_times, file);
    RS.files = as_cells(D.files);
    RS.held = as_cells(D.held);
    seen = {};
    for k = 1:numel(RS.files)
        F = RS.files{k};
        seen = add_session(seen, F, file);
        check_times(F, file);   % both stim times, and the input mask = their maximum
        rows = as_cells(F.analyses);
        names = cellfun(@(r) char(r.analysis), rows, 'UniformOutput', false);
        if numel(unique(names)) ~= numel(names)
            error('night6:recoveryStartFile', '%s: %s lists an analysis twice', file, F.session);
        end
        for j = 1:numel(rows)
            k0 = rows{j}.start_sample0;
            if ~is_sample(k0)
                error('night6:recoveryStartFile', ['%s: %s/%s start_sample0 must be one ' ...
                      'integer >= 0'], file, F.session, names{j});
            end
        end
        RS.files{k}.analyses = rows;
        if ~isfield(F, 'cuts') || isempty(F.cuts)
            error('night6:recoveryStartFile', ['%s: %s has no cuts (RULING 2026-10-09 ' ...
                  'item 6: one per owner, output and class)'], file, F.session);
        end
        cuts = as_cells(F.cuts);
        ids = cellfun(@(c) char(c.cut), cuts, 'UniformOutput', false);
        if numel(unique(ids)) ~= numel(ids)
            error('night6:recoveryStartFile', '%s: %s lists a cut twice', file, F.session);
        end
        for j = 1:numel(cuts)
            if ~is_sample(cuts{j}.start_sample0)
                error('night6:recoveryStartFile', ['%s: %s/%s start_sample0 must be one ' ...
                      'integer >= 0'], file, F.session, ids{j});
            end
        end
        RS.files{k}.cuts = cuts;
    end
    for k = 1:numel(RS.held)
        seen = add_session(seen, RS.held{k}, file);
    end
end

function check_times(F, file)
% RULING 2026-10-09 (i) 3 (c): stim-off is two times. Both are required, the electrical
% settling and the mechanical end are exact samples, and the input mask point Night 6
% masks through is their maximum, to the sample; no measured start or cut precedes it.
    for g = {'electrical_settle_sample0', 'mechanical_end_sample0', 'input_mask_sample0'}
        if ~isfield(F, g{1}) || ~is_sample(F.(g{1}))
            error('night6:recoveryStartFile', '%s: %s %s must be one integer >= 0', ...
                  file, F.session, g{1});
        end
    end
    for g = {'electrical_end_s', 'mechanical_end_s', 'electrical_settle_s', 'input_mask_s'}
        if ~isfield(F, g{1}) || ~isnumeric(F.(g{1})) || ~isscalar(F.(g{1})) ...
                || ~isfinite(F.(g{1}))
            error('night6:recoveryStartFile', ['%s: %s has no %s (both stim times are ' ...
                  'required, RULING 2026-10-09 (i) 3 (c))'], file, F.session, g{1});
        end
    end
    if ~isfield(F, 'fs') || ~isnumeric(F.fs) || ~isscalar(F.fs) || ~isfinite(F.fs) ...
            || F.fs <= 0
        error('night6:recoveryStartFile', '%s: %s has no positive finite fs', file, F.session);
    end
    k = double(F.input_mask_sample0);
    if k ~= max(double(F.electrical_settle_sample0), double(F.mechanical_end_sample0))
        error('night6:recoveryStartFile', ['%s: %s input_mask_sample0 %d is not ' ...
              'max(electrical_settle_sample0 %d, mechanical_end_sample0 %d)'], file, ...
              F.session, k, F.electrical_settle_sample0, F.mechanical_end_sample0);
    end
    % each sample is the first sample at or after its seconds field (review 8 finding 4;
    % night6_first_sample = Python extent.grid.first_sample_at_or_after, the one conversion)
    for g = {'electrical_settle_sample0', 'electrical_settle_s'; ...
             'mechanical_end_sample0', 'mechanical_end_s'; ...
             'input_mask_sample0', 'input_mask_s'}'
        want = night6_first_sample(F.(g{2}), F.fs);
        if double(F.(g{1})) ~= want
            error('night6:recoveryStartFile', ['%s: %s %s %d is not the first sample at ' ...
                  'or after %s %.17g s at fs %.17g (%d; night6_first_sample)'], file, ...
                  F.session, g{1}, F.(g{1}), g{2}, F.(g{2}), F.fs, want);
        end
    end
    if F.electrical_settle_s < F.electrical_end_s
        error('night6:recoveryStartFile', ['%s: %s electrical_settle_s %.17g precedes ' ...
              'electrical_end_s %.17g: the settling is measured from the electrical end'], ...
              file, F.session, F.electrical_settle_s, F.electrical_end_s);
    end
    rows = as_cells(F.analyses);
    if isfield(F, 'cuts'), rows = [rows, as_cells(F.cuts)]; end
    for j = 1:numel(rows)
        r = rows{j};
        if isfield(r, 'basis') && any(strcmp(r.basis, {'input_mask_plus_own_settling', ...
                                                       'input_mask_plus_class_reach'})) ...
                && isnumeric(r.start_sample0) && double(r.start_sample0) < k
            error('night6:recoveryStartFile', ['%s: %s has a measured start (sample %d) ' ...
                  'before its input mask sample %d'], file, F.session, r.start_sample0, k);
        end
    end
end

function check_sources(list, file)
% Every processing_new file the table and the output time map cite (Python
% extent.recovery_start.SOURCE_FILES) must be the file MATLAB resolves now, byte for
% byte: a start or a stamp convention read from one version of her code is never applied
% to the outputs of another. Rows are {file, sha256}, never JSON keys (invariant 22).
    rows = as_cells(list);
    if isempty(rows)
        error('night6:recoveryStartFile', '%s: source_files is empty', file);
    end
    for k = 1:numel(rows)
        r = rows{k};
        if ~isstruct(r) || ~all(isfield(r, {'file', 'sha256'})) || ~ischar(r.file) ...
                || ~ischar(r.sha256)
            error('night6:recoveryStartFile', '%s: source_files row %d is not {file, sha256}', ...
                  file, k);
        end
        p = which(r.file);
        if isempty(p) || ~isfile(p)
            error('night6:recoveryStartSource', ['%s: cited by %s, is not on the MATLAB ' ...
                  'path, so the code the starts were read from cannot be checked'], r.file, file);
        end
        h = night6_sha256_file(p);
        if ~strcmp(h, lower(r.sha256))
            error('night6:recoveryStartSource', ['%s resolves to %s with sha256 %s, but %s ' ...
                  'was computed from sha256 %s: her code changed since the starts and the ' ...
                  'output time map were read from it'], r.file, p, h, file, r.sha256);
        end
    end
end

function Ed = check_edge_settling(D, file)
% The starts file's edge settlings must be the ones Night 6 applies now: its recorded
% hash equals night6_edge_settling()'s sha256 (the raw bytes of edge_settling.json, which
% Python's tolerance.edge_settling_text() reproduces) and its decoded content equals the
% declaration's. A file written under other settlings is stale and refused by name.
    Ed = night6_edge_settling();
    h = D.edge_settling_sha256;
    if ~ischar(h) || ~strcmpi(h, Ed.sha256)
        error('night6:recoveryStartEdgeSettling', ['%s was derived under edge settlings ' ...
              'with sha256 %s, but night6_edge_settling (%s) has sha256 %s: the starts ' ...
              'file is stale - regenerate it'], file, char(string(h)), Ed.file, Ed.sha256);
    end
    if ~isequal(D.edge_settling, rmfield(Ed, {'file', 'sha256'}))
        error('night6:recoveryStartEdgeSettling', ['%s: its edge_settling content differs ' ...
              'from night6_edge_settling (%s) although the hash matches'], file, Ed.file);
    end
end

function tf = is_sample(k)
    tf = isnumeric(k) && isscalar(k) && k == fix(k) && k >= 0;
end

function T = output_times(D, file)
% The output time map as cells of structs; refused by name when malformed.
    if ~isstruct(D) || ~all(isfield(D, {'files', 'vars', 'xchan_delay_params', ...
                                        'marker_variable'})) ...
            || ~isvarname(D.marker_variable)
        error('night6:recoveryStartFile', ['%s: output_times has no files/vars/' ...
              'xchan_delay_params/marker_variable'], file);
    end
    T = struct('files', {as_cells(D.files)}, 'vars', {as_cells(D.vars)}, ...
               'xchanDelayParams', {cellstr(D.xchan_delay_params(:)')}, ...
               'markerVariable', D.marker_variable, 'conventions', struct(), ...
               'trimClasses', struct(), 'ruling', '', 'suffix', '_validFraction');
    if isfield(D, 'conventions') && isstruct(D.conventions)
        T.conventions = D.conventions;   % name -> meaning (identifiers: jsondecode keeps them)
    end
    if isfield(D, 'trim_classes') && isstruct(D.trim_classes)
        T.trimClasses = D.trim_classes;  % valid_only / filled_or_filtered -> meaning
    end
    if isfield(D, 'ruling'), T.ruling = D.ruling; end
    if isfield(D, 'valid_fraction_suffix'), T.suffix = D.valid_fraction_suffix; end
    for k = 1:numel(T.vars)
        v = T.vars{k};
        if ~all(isfield(v, {'file', 'path', 'role', 'owner'}))
            error('night6:recoveryStartFile', '%s: output_times var %d is malformed', file, k);
        end
        if strcmp(v.role, 'trim') && ~all(isfield(v, {'stamp', 'convention', 'action'}))
            error('night6:recoveryStartFile', '%s: %s/%s has no stamp, convention or action', ...
                  file, v.file, v.path);
        end
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
