function E = night6_edge_settling(f)
% NIGHT6_EDGE_SETTLING  The measured edge settlings Night 6 applies, with their provenance.
%
%   E = night6_edge_settling()        reads edge_settling.json beside this file
%   E = night6_edge_settling(file)
%
% RULING 2026-10-09 (c) 1-2. The declaration is edge_settling.json, a copy of Python's
% gems_blanking_v2.extent.tolerance.edge_settling_record() (THE construction site,
% EDGE_SETTLING; a test holds the two equal), so the spike edge pad, the mmc pad and the
% measurement files they rest on are stated once:
%   E.spikes.edge_buffer_ms  P.edgeBufferMs for v2 (10.5: the measured 10.282 ms zero-phase
%                            edge settling - step1 7.782 ms + step4 2.5 ms - rounded up to
%                            0.5 ms, invariant 19)
%   E.mmc.pad_s              the mmc padding at a blanked edge (1.5 s; 1.1594 s measured)
%   E.<consumer>.files       [{file, sha256}] of the measurement each figure came from
% plus E.file and E.sha256 (this declaration's path and raw-byte SHA-256), so a record
% names exactly what it applied.
%
% Refused by name ('night6:edgeSettling'): an unreadable file, a missing field, and a pad
% smaller than the settling it covers or not rounded up to 0.5 ms (a pad that understates
% the measurement silently loses coverage - invariant 19).
    if nargin < 1 || isempty(f)
        f = fullfile(fileparts(mfilename('fullpath')), 'edge_settling.json');
    end
    if ~isfile(f)
        error('night6:edgeSettling', 'the edge-settling declaration %s does not exist', f);
    end
    E = jsondecode(fileread(f));
    need = struct('spikes', {{'filter_s', 'before_gap_s', 'after_gap_s', 'total_s', ...
                              'pad_s', 'edge_buffer_ms', 'ruling', 'files'}}, ...
                  'mmc', {{'filter_s', 'total_s', 'pad_s', 'ruling', 'files'}});
    for c = fieldnames(need)'
        if ~isfield(E, c{1})
            error('night6:edgeSettling', '%s has no %s entry', f, c{1});
        end
        for k = need.(c{1})
            if ~isfield(E.(c{1}), k{1})
                error('night6:edgeSettling', '%s: %s.%s is absent', f, c{1}, k{1});
            end
        end
        files = E.(c{1}).files;
        if isempty(files) || ~all(arrayfun(@(r) numel(char(r.sha256)) == 64, files))
            error('night6:edgeSettling', '%s: %s names no measurement file with a SHA-256', ...
                  f, c{1});
        end
    end
    ms = E.spikes.edge_buffer_ms;
    if ~(isnumeric(ms) && isscalar(ms) && ms >= 1e3 * E.spikes.total_s - 1e-9 ...
            && abs(ms * 2 - round(ms * 2)) < 1e-9 && ms - 1e3 * E.spikes.total_s < 0.5)
        error('night6:edgeSettling', ['%s: spike edge pad %g ms is not the measured %g ms ' ...
              'rounded up to 0.5 ms'], f, ms, 1e3 * E.spikes.total_s);
    end
    if abs(1e3 * E.spikes.pad_s - ms) > 1e-9
        error('night6:edgeSettling', '%s: spikes.pad_s %g s and edge_buffer_ms %g disagree', ...
              f, E.spikes.pad_s, ms);
    end
    if ~(E.mmc.pad_s >= E.mmc.total_s)
        error('night6:edgeSettling', '%s: mmc pad %g s is under its measured %g s', f, ...
              E.mmc.pad_s, E.mmc.total_s);
    end
    E.file = f;
    E.sha256 = night6_sha256_file(f);
end
