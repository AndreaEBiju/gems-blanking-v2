function [lead, rec] = night6_recovery_lead_in(RS, session, condition, i0, n, fs, consumers)
% NIGHT6_RECOVERY_LEAD_IN  How many leading rows of this epoch each consumer must mask.
%
%   [lead, rec] = night6_recovery_lead_in(RS, session, condition, i0, n, fs, consumers)
%
%   RS         night6_recovery_start(file), or [] when no file was declared
%   session    the recording (the mask provenance's recording)
%   condition  the mask provenance's condition
%   i0, n      the epoch: file samples i0+1 .. i0+n (1-based), i0 = epochStartSample0
%   fs         the epoch's sample rate
%   consumers  the consumers this epoch will run (status 'to_run')
%
% RULING 2026-10-08 (k) 2: Night 6 runs every stim_rec recovery epoch from its own start
% (132 s), and TRIMS each analysis whose own start (stim-off + electrical settling + its
% own settling) is later. TRIM = MASK THE INPUT before the start: the consumer's leading
% epoch rows 1 .. k0 - i0 are added to its mask as NaN, where k0 is the analysis's start
% as a 0-based file sample (rows are 1-based into the epoch, so row r is file sample
% i0 + r - 1 0-based, and row k0 - i0 is the last sample before the start). Masking, not
% dropping outputs timed before the start, because her analyses compute epoch-wide
% statistics (spike sigma, slow-wave detrend, average rates) that no output time can
% trim: only data the analysis never sees cannot leak into an output at or after the
% start. The added span is NaN like any motion span (invariant 1) and honoured by her
% functions exactly as one.
%
% lead.<consumer> = number of leading rows to mask (0 .. n). A start before the epoch
% start is not an error: the analysis runs from the epoch start and its record says
% "early part deferred to the add-on". A start at the epoch start (an analysis whose
% settling is unknown keeps 132 s, labelled fixed_132s_settling_unknown) masks nothing.
%
% Refused by name - never a silent fallback:
%   night6:recoveryStarts          a stim_recovery epoch and no declared file
%   night6:recoveryStartHeld       the file is held (stim edges undetected, or never settles)
%   night6:recoveryStartMissing    no row for the file, or none for a consumer that runs
%   night6:recoveryStartFs         the row's fs is not the epoch's
%   night6:recoveryStartCondition  a row for a recording that is not stim_recovery
    lead = struct();
    rec = struct('applies', false, 'condition', char(condition));
    if ~isempty(RS)
        rec.file = RS.file;
        rec.sha256 = RS.sha256;
    end
    if ~strcmp(condition, 'stim_recovery')
        if ~isempty(RS) && (~isempty(find_session(RS.files, session)) ...
                            || ~isempty(find_session(RS.held, session)))
            error('night6:recoveryStartCondition', ['%s has a recovery start in %s but its ' ...
                  'condition is ''%s'', not stim_recovery'], session, RS.file, char(condition));
        end
        rec.reason = 'not a stim_recovery recording: no recovery start applies';
        return
    end
    if isempty(RS)
        error('night6:recoveryStarts', ['%s is stim_recovery: its analyses need their ' ...
              'recovery starts (RULING 2026-10-08 (k) 2), declared with RecoveryStarts'], session);
    end
    H = find_session(RS.held, session);
    if ~isempty(H)
        error('night6:recoveryStartHeld', '%s is held in %s: %s', session, RS.file, ...
              char(H.basis));
    end
    F = find_session(RS.files, session);
    if isempty(F)
        error('night6:recoveryStartMissing', 'no recovery start for %s in %s', session, RS.file);
    end
    if abs(double(F.fs) - fs) > 1e-9
        error('night6:recoveryStartFs', '%s: recovery start at fs %.6f, epoch at fs %.6f', ...
              session, double(F.fs), fs);
    end
    rec.applies = true;
    rec.epoch_start_sample0 = i0;
    for f = {'stim_off_s', 'electrical_settle_s', 'electrical_s'}
        if isfield(F, f{1}), rec.(f{1}) = F.(f{1}); end
    end
    rec.consumers = struct();
    names = cellfun(@(r) char(r.analysis), F.analyses, 'UniformOutput', false);
    for c = consumers(:)'
        j = find(strcmp(names, c{1}));
        if numel(j) ~= 1
            error('night6:recoveryStartMissing', ['%s: no recovery start for analysis %s ' ...
                  'in %s - it is never run from the epoch start by default'], session, c{1}, ...
                  RS.file);
        end
        r = F.analyses{j};
        k0 = double(r.start_sample0);
        rows = min(n, max(0, k0 - i0));
        e = struct('start_s', r.start_s, 'start_sample0', k0, 'basis', char(r.basis), ...
                   'source', char(r.source), 'trimmed_rows', rows);
        for g = {'own_settling_s', 'binding_output', 'missing_settling'}
            if isfield(r, g{1}), e.(g{1}) = r.(g{1}); end
        end
        if k0 > i0
            e.status = 'trimmed';
            e.rule = sprintf(['input masked on epoch rows 1..%d (file samples %d..%d, ' ...
                              '0-based): the analysis starts at its own start'], rows, i0, ...
                             i0 + rows - 1);
        elseif k0 == i0
            e.status = 'at_epoch_start';
        else
            e.status = 'early part deferred to add-on';
            e.rule = sprintf(['own start %d samples before the epoch start: runs from the ' ...
                              'epoch start; the early recovery is appended by the add-on'], i0 - k0);
        end
        rec.consumers.(c{1}) = e;
        lead.(c{1}) = rows;
    end
end

function F = find_session(list, session)
    F = [];
    for k = 1:numel(list)
        if strcmp(list{k}.session, session)
            F = list{k};
            return
        end
    end
end
