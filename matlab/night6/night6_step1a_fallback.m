function F = night6_step1a_fallback(file)
% NIGHT6_STEP1A_FALLBACK  Read the declared step1a fallback list (RULING 2026-10-08 (j) 1).
%
%   F = night6_step1a_fallback(file)
%
% file is UTF-8 JSON (default list: step1a_fallback.json beside this file, empty):
%   { "schema": 1, "ruling": "...", "note": "...",
%     "entries": [ {"animal": "B", "cuff": "L", "reason": "...",
%                   "decided": "2026-10-11", "source": "..."} ] }
% An entry names one animal x cuff whose spike consumer falls back to Andrea's
% step1a_blank_cardiac inside process_dataset_v2: only where the peri-R test of ruling
% (i) 3(b) found an excess AND ruling 2026-10-07 (e) classified it as leak. Firing is
% kept, never listed.
%
% Refused, by name ('night6:fallback*'): an unknown key at either level, a schema other
% than 1, an entry without animal, cuff or reason, a cuff other than L or R, an animal
% that is not a single upper-case token, and a duplicate animal x cuff. Nothing is
% defaulted: a list that cannot be read stops the run.
%
% F.file, F.sha256 (of the file's raw bytes, night6_sha256_file - never of the decoded
% text, which would hide a CRLF, BOM or encoding difference), F.entries (cell of structs,
% possibly empty), F.keys (cellstr "<animal>|<cuff>").
    txt = fileread(file, 'Encoding', 'UTF-8');
    J = jsondecode(txt);
    top = {'schema', 'ruling', 'note', 'entries'};
    bad = setdiff(fieldnames(J), top);
    if ~isempty(bad)
        error('night6:fallbackKey', '%s: unknown key(s) [%s]; allowed [%s]', file, ...
              strjoin(bad, ' '), strjoin(top, ' '));
    end
    if ~isfield(J, 'schema') || ~isequal(J.schema, 1)
        error('night6:fallbackSchema', '%s: schema must be 1', file);
    end
    if ~isfield(J, 'entries')
        error('night6:fallbackKey', '%s: no "entries" list', file);
    end
    E = J.entries;
    if isempty(E)
        E = {};
    elseif isstruct(E)
        E = num2cell(E(:)');   % entries with different keys decode as a cell
    end
    allowed = {'animal', 'cuff', 'reason', 'decided', 'source'};
    keys = cell(1, numel(E));
    for k = 1:numel(E)
        e = E{k};
        bad = setdiff(fieldnames(e), allowed);
        if ~isempty(bad)
            error('night6:fallbackKey', '%s: entry %d has unknown key(s) [%s]; allowed [%s]', ...
                  file, k, strjoin(bad, ' '), strjoin(allowed, ' '));
        end
        for req = {'animal', 'cuff', 'reason'}
            if ~isfield(e, req{1}) || isempty(e.(req{1}))
                error('night6:fallbackEntry', '%s: entry %d has no %s', file, k, req{1});
            end
        end
        if ~any(strcmp(e.cuff, {'L', 'R'}))
            error('night6:fallbackEntry', '%s: entry %d cuff ''%s'' is not L or R', ...
                  file, k, char(e.cuff));
        end
        if ~ischar(e.animal) || isempty(regexp(e.animal, '^[A-Z]+$', 'once'))
            error('night6:fallbackEntry', '%s: entry %d animal ''%s'' is not an animal key', ...
                  file, k, char(string(e.animal)));
        end
        keys{k} = sprintf('%s|%s', e.animal, e.cuff);
    end
    if numel(unique(keys)) ~= numel(keys)
        error('night6:fallbackDuplicate', '%s: an animal x cuff is listed twice', file);
    end
    F = struct('file', char(file), 'sha256', night6_sha256_file(file), ...
               'entries', {E}, 'keys', {keys});
end
