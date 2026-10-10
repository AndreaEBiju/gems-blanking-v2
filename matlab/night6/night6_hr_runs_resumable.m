function [tf, why] = night6_hr_runs_resumable(R)
% NIGHT6_HR_RUNS_RESUMABLE  Whether a record's HR runs may be resumed. ONE site.
%
%   [tf, why] = night6_hr_runs_resumable(R)
%
% Review 7 finding 5 (RULING 2026-10-09 (i) 4): a complete record is resumed only when every
% HR_BR_HRVAnalysis_beats run in it served exactly ONE consumer (hrv or breathing) and
% records outputs_used, the record night6_hr_outputs writes. A record from before (i) 4 - one
% shared HR call for hrv and breathing, or no outputs_used - is rerun, never reported done.
% A record with no runs list is not resumable either. R is a jsondecode'd night6_record.json
% (runs decode to a struct array or a cell array of structs; consumers to a cell or a char).
% WHY names the first reason, '' when TF is true.
    tf = false;
    why = '';
    if ~isstruct(R) || ~isfield(R, 'runs')
        why = 'the record has no runs list';
        return
    end
    runs = R.runs;
    if isempty(runs) && ~iscell(runs) && ~isstruct(runs)
        runs = {};   % an empty JSON list decodes to []
    end
    if isstruct(runs)
        runs = num2cell(runs);
    elseif ~iscell(runs)
        why = 'the record''s runs are neither a struct array nor a cell array';
        return
    end
    for k = 1:numel(runs)
        r = runs{k};
        if ~isstruct(r) || ~isfield(r, 'call')
            why = sprintf('run %d has no call', k);
            return
        end
        if ~strcmp(r.call, 'HR_BR_HRVAnalysis_beats')
            continue
        end
        if ~isfield(r, 'consumers')
            why = sprintf('HR run %d names no consumers', k);
            return
        end
        c = r.consumers;
        if ischar(c) || isstring(c)
            c = cellstr(c);
        end
        if ~iscell(c) || numel(c) ~= 1 || ~ismember(c{1}, {'hrv', 'breathing'})
            why = sprintf(['HR run %d serves %d consumer(s); (i) 4 runs hrv and breathing ' ...
                           'as two calls, one consumer each'], k, numel(c));
            return
        end
        if ~isfield(r, 'outputs_used') || ~isstruct(r.outputs_used) ...
                || ~isfield(r.outputs_used, 'consumer') ...
                || ~strcmp(r.outputs_used.consumer, c{1})
            why = sprintf('HR run %d (%s) records no outputs_used for its consumer', k, c{1});
            return
        end
    end
    tf = true;
end
