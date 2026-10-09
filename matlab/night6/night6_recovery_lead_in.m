function [lead, rec] = night6_recovery_lead_in(RS, session, condition, i0, n, fs, consumers, mode)
% NIGHT6_RECOVERY_LEAD_IN  How many leading rows of this epoch each consumer must mask.
%
%   [lead, rec] = night6_recovery_lead_in(RS, session, condition, i0, n, fs, consumers, mode)
%
%   RS         night6_recovery_start(file), or [] when no file was declared
%   session    the recording (the mask provenance's recording)
%   condition  the mask provenance's condition
%   i0, n      the epoch: file samples i0+1 .. i0+n (1-based), i0 = epochStartSample0
%   fs         the epoch's sample rate
%   consumers  the consumers this epoch will run (status 'to_run')
%   mode       the declared trim mode (night6_trim_modes; refused by name otherwise)
%
% RULING 2026-10-08 (k) 2 and RULING 2026-10-09 item 6 (mode (B)): Night 6 runs every
% stim_rec recovery epoch from its own start (132 s). The file's electrical settling ke,
% each analysis's own start k0 and each CUT's point kc (owner.output.class) are exact
% 0-based FILE samples. Rows are 1-based into the epoch, so row r is file sample
% i0 + r - 1 (0-based) and the rows before file sample k are 1 .. k - i0.
%
%   EVERY consumer's input rows 1 .. ke - i0 are NaN (one point for all, so her
%   epoch-wide statistics see no unsettled data); each output variable stamped before
%   row kc - i0 + 1 of its cut is dropped or flagged after the call
%   (night6_recovery_trim_outputs, by the output time map's class and cut).
%
% The added span is NaN like any motion span (invariant 1) and honoured by her functions
% exactly as one. Slow wave's 15 s edge guard DOES act at it: Night 6 passes each run's
% masked spans, this lead-in included, as blankIdx (RULING 2026-10-09 (c) 6). HR's does
% not: HR gets blankIdx = [], and its edge mask sits only at blankIdx and the array ends.
%
% lead.<consumer> = number of leading INPUT rows to mask (0 .. n);
% rec.cuts{} = cut, owner, output_key, trim_class, basis, start_s, start_sample0 and
% output_rows_before_start (0 .. n) of EVERY cut in the file's row (the trimmer cuts
% each variable at its own); rec.analyses.<analysis> = each analysis's own start (the
% latest of its cuts' full reach, for the record and the add-on). A cut or a start before
% the epoch start is not an error: nothing is cut there and the record says "early part
% deferred to the add-on".
%
% Refused by name - never a silent fallback:
%   night6:recoveryTrimMode        no mode, an unknown one, or a withdrawn one ((A))
%   night6:recoveryStarts          a stim_recovery epoch and no declared file
%   night6:recoveryStartHeld       the file is held (stim edges undetected, or never settles)
%   night6:recoveryStartMissing    no row for the file, or none for a consumer that runs
%   night6:recoveryStartFs         the row's fs is not the epoch's
%   night6:recoveryStartCondition  a row for a recording that is not stim_recovery
    mode = night6_check_trim_mode(mode);
    lead = struct();
    rec = struct('applies', false, 'condition', char(condition), 'mode', mode);
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
    for f = {'stim_off_s', 'electrical_settle_s', 'electrical_settle_sample0', 'electrical_s'}
        if isfield(F, f{1}), rec.(f{1}) = F.(f{1}); end
    end
    ke = double(F.electrical_settle_sample0);
    rows = min(n, max(0, ke - i0));
    rec.rule = sprintf(['input of every analysis masked on epoch rows 1..%d (before the ' ...
                        'electrical settling, file sample %d 0-based); each output variable ' ...
                        'stamped before its own cut dropped or flagged (RULING 2026-10-09 ' ...
                        'item 6)'], rows, ke);
    rec.epoch_scalars = ['computed over [electrical settling, epoch end]: they have no ' ...
                         'time to trim by (whole-epoch averages of trimmed series are ' ...
                         'recomputed from the kept values)'];
    rec.cuts = cellfun(@(c) struct('cut', char(c.cut), 'owner', char(c.owner), ...
        'output_key', char(c.output_key), 'trim_class', char(c.trim_class), ...
        'basis', char(c.basis), 'start_s', double(c.start_s), ...
        'start_sample0', double(c.start_sample0), ...
        'output_rows_before_start', min(n, max(0, double(c.start_sample0) - i0))), ...
        F.cuts, 'UniformOutput', false);
    rec.consumers = struct();
    names = cellfun(@(r) char(r.analysis), F.analyses, 'UniformOutput', false);
    rec.analyses = struct();
    for j = 1:numel(F.analyses)
        r = F.analyses{j};
        if ~isvarname(names{j})
            error('night6:recoveryStartMissing', '%s: analysis name %s is not an identifier', ...
                  session, names{j});
        end
        k0 = double(r.start_sample0);
        rec.analyses.(names{j}) = struct('start_s', r.start_s, 'start_sample0', k0, ...
            'output_rows_before_start', min(n, max(0, k0 - i0)));
    end
    for c = consumers(:)'
        j = find(strcmp(names, c{1}));
        if numel(j) ~= 1
            error('night6:recoveryStartMissing', ['%s: no recovery start for analysis %s ' ...
                  'in %s - it is never run from the epoch start by default'], session, c{1}, ...
                  RS.file);
        end
        r = F.analyses{j};
        k0 = double(r.start_sample0);
        e = struct('start_s', r.start_s, 'start_sample0', k0, 'basis', char(r.basis), ...
                   'source', char(r.source), 'trimmed_rows', rows, ...
                   'output_rows_before_start', min(n, max(0, k0 - i0)));
        for g = {'own_settling_s', 'binding_output', 'missing_settling'}
            if isfield(r, g{1}), e.(g{1}) = r.(g{1}); end
        end
        if ke > i0
            e.status = 'trimmed';
            e.rule = sprintf('input masked on epoch rows 1..%d (file samples %d..%d, 0-based)', ...
                             rows, i0, i0 + rows - 1);
        elseif ke == i0
            e.status = 'at_epoch_start';
        else
            e.status = 'early part deferred to add-on';
            e.rule = sprintf(['input mask end %d samples before the epoch start: runs from ' ...
                              'the epoch start; the early recovery is appended by the add-on'], ...
                             i0 - ke);
        end
        mine = cellfun(@(x) strcmp(x.owner, c{1}), rec.cuts);
        if any(cellfun(@(x) x.start_sample0 > i0, rec.cuts(mine)))
            e.output_status = 'outputs before their own cuts dropped or flagged, per variable';
        else
            e.output_status = 'every cut at or before the epoch start: no output dropped';
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
