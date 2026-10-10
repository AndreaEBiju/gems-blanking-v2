function summary = night6_batch(listFile, nWorkers, varargin)
% NIGHT6_BATCH  Night 6 driver: night6_run_recording over a list of mask folders.
%
%   summary = night6_batch(listFile, nWorkers)
%   summary = night6_batch(listFile, nWorkers, 'DryRun', true)
%   summary = night6_batch(listFile, nWorkers, 'KeepInputs', true)   (keep each epoch's
%             <base>_spikes_in.mat etc., e.g. for the pilot's v1/v2/ECAP validation)
%
% listFile is UTF-8 JSON:
%   { "gems_root": "<store root>",          (local path; never shared)
%     "units": "V",                         (declared for the cohort, invariants 14/24)
%     "out_root": "<output folder>",
%     "processing_new": "<path>",           (optional; default ../../../processing_new)
%     "pilot_run": true,                    (optional: declares a PILOT run, which must name
%     "pilot_root": "<path>",                its folder; out_root must lie inside it, and it
%                                            must not be or sit under a junction or symbolic
%                                            link - night6_pilot_root. Only a pilot run reads
%                                            STAND-IN tolerance masks or takes beats_root)
%     "beats_root": "<path>",               (optional, PILOT RUNS ONLY: the beats files
%                                            resolve here instead of gems_root -
%                                            night6_run_recording BeatsRoot)
%     "label": "free text",                 (optional; copied into every record)
%     "recovery_starts": "<path>",          (RULING 2026-10-08 (k) 2; required for any
%                                            stim_recovery recording - see RecoveryStarts)
%     "recovery_trim_mode": "<mode>",       (REQUIRED, no default: mode (B),
%                                            mask_to_electrical_drop_outputs - see
%                                            RecoveryTrimMode; (A) is refused by name)
%     "slow_wave_rate": "<rate>",           (REQUIRED, no default: 'full' | 'decimated78',
%                                            RULING 2026-10-09 (c) 6; night6_slow_wave_rates)
%     "recordings": [ {"mask_folder": "data/A/<session>/masks/<model-id>",
%                      "meta_file": "..."} ] }   (meta_file optional)
% A relative path is POSIX and resolves against gems_root (cross-platform rule 2).
%
% nWorkers: 0 runs in this session, single-threaded (maxNumCompThreads(1), restored
% after) so its timings match a pool worker's (invariant 36); >= 1 opens a pool of
% exactly that many on an IN-SESSION copy of the 'Processes' cluster whose NumWorkers
% is raised if needed - the persistent profile is never saved (invariant 39).
% Resumable: an epoch whose night6_record.json says "complete" is skipped; a failed
% recording does not stop the batch. Writes <out_root>/night6_batch_<time>.json.
%
% Refused AT BATCH START, by name, before any recording is loaded (RULING 2026-10-08
% (k) 2; review of be402a1): a missing, unknown or withdrawn recovery_trim_mode
% ('night6:recoveryTrimMode'; (A) names RULING 2026-10-09 item 6); a missing or unknown
% slow_wave_rate ('night6:slowWaveRate', RULING 2026-10-09 (c) 6); any listed mask folder holding a stim_recovery epoch
% when no recovery_starts is declared ('night6:recoveryStarts' - the conditions are read
% from the mask files' provenance only, never from the signals); and an unreadable or
% malformed recovery_starts file (night6_recovery_start, read once here); half a pilot
% declaration ('night6:pilotRun'), a pilot_root that does not exist, is or sits under a
% junction or symbolic link, or does not hold out_root ('night6:pilotRoot'); and a
% beats_root in a list that declares no pilot run ('night6:beatsRoot'; review 2026-10-10
% fix 3). pilot_root and beats_root resolve like recovery_starts (relative: POSIX, against
% gems_root).
    ip = inputParser;
    ip.addParameter('DryRun', false);
    ip.addParameter('Figures', false);
    ip.addParameter('Force', false);
    ip.addParameter('KeepInputs', false, @(x) islogical(x) || isnumeric(x));
    ip.parse(varargin{:});
    opt = ip.Results;

    L = jsondecode(fileread(listFile));
    mode = '';
    if isfield(L, 'recovery_trim_mode'), mode = L.recovery_trim_mode; end
    mode = night6_check_trim_mode(mode);   % (k) 2: required, refused by name here
    swRate = '';
    if isfield(L, 'slow_wave_rate'), swRate = L.slow_wave_rate; end
    [~, SW] = night6_slow_wave_rates(swRate);   % (c) 6: required, refused by name here
    here = fileparts(mfilename('fullpath'));
    pnew = fullfile(here, '..', '..', '..', 'processing_new');
    if isfield(L, 'processing_new'), pnew = L.processing_new; end
    addpath(pnew); addpath(here);
    label = '';
    if isfield(L, 'label'), label = L.label; end
    [st, out] = system(sprintf('git -C "%s" rev-parse HEAD', here));
    if st ~= 0, error('night6:commit', 'cannot read the code commit: %s', out); end
    commit = strtrim(out);

    recs = L.recordings;
    if isstruct(recs), recs = num2cell(recs); end
    n = numel(recs);
    folders = cell(1, n); metas = cell(1, n);
    for i = 1:n
        folders{i} = resolve(L.gems_root, recs{i}.mask_folder);
        metas{i} = '';
        if isfield(recs{i}, 'meta_file'), metas{i} = resolve(L.gems_root, recs{i}.meta_file); end
    end
    starts = '';
    if isfield(L, 'recovery_starts'), starts = resolve(L.gems_root, L.recovery_starts); end
    pilotRoot = '';   % review 2026-10-10 fixes 1-3: a pilot run is declared, never assumed
    if isfield(L, 'pilot_run') || isfield(L, 'pilot_root')
        if ~(isfield(L, 'pilot_run') && isequal(L.pilot_run, true)) ...
                || ~isfield(L, 'pilot_root') || isempty(L.pilot_root)
            error('night6:pilotRun', ['a pilot run is declared by "pilot_run": true together ' ...
                  'with "pilot_root": <folder>; this list has only part of that']);
        end
        pilotRoot = night6_pilot_root(resolve(L.gems_root, L.pilot_root), L.out_root);
    end
    broot = '';
    if isfield(L, 'beats_root')
        if isempty(pilotRoot)
            error('night6:beatsRoot', ['beats_root is for pilot runs only (review 2026-10-10 ' ...
                  'fix 3): this list declares no pilot run ("pilot_run": true, "pilot_root")']);
        end
        broot = resolve(L.gems_root, L.beats_root);
    end
    preflight_recovery(folders, starts);   % (k) 2: before any recording is loaded
    args = {'GemsRoot', L.gems_root, 'Units', L.units, 'OutRoot', L.out_root, ...
            'CodeCommit', commit, 'Label', label, 'DryRun', opt.DryRun, ...
            'Figures', opt.Figures, 'Force', opt.Force, 'KeepInputs', logical(opt.KeepInputs), ...
            'RecoveryStarts', starts, 'BeatsRoot', broot, 'PilotRoot', pilotRoot, ...
            'RecoveryTrimMode', mode, 'SlowWaveRate', SW.name};
    status = cell(1, n); wall = zeros(1, n);
    t0 = tic;
    fprintf('[night6] %d recording(s), %d worker(s), commit %s\n', n, nWorkers, commit);
    if nWorkers == 0
        threads = maxNumCompThreads(1);
        restore = onCleanup(@() maxNumCompThreads(threads)); %#ok<NASGU>
        for i = 1:n
            [status{i}, wall(i)] = one(folders{i}, metas{i}, args);
        end
    else
        c = parcluster('Processes');
        if c.NumWorkers < nWorkers, c.NumWorkers = nWorkers; end   % this session only
        pool = gcp('nocreate');
        if ~isempty(pool) && pool.NumWorkers ~= nWorkers, delete(pool); pool = []; end
        if isempty(pool), pool = parpool(c, nWorkers); end %#ok<NASGU>
        pctRunOnAll(sprintf('addpath(''%s''); addpath(''%s'');', pnew, here));
        parfor i = 1:n
            [status{i}, wall(i)] = one(folders{i}, metas{i}, args);
        end
    end
    summary = struct('list_file', listFile, 'n', n, 'workers', nWorkers, 'commit', commit, ...
                     'wall_s', toc(t0), 'label', label, 'recovery_trim_mode', mode, ...
                     'slow_wave_rate', SW.name, 'pilot_root', pilotRoot, 'beats_root', broot);
    summary.recordings = cellfun(@(f, s, w) struct('mask_folder', f, 'status', s, 'wall_s', w), ...
                                 folders, status, num2cell(wall), 'UniformOutput', false);
    summary.n_ok = nnz(strcmp(status, 'ok'));
    stamp = char(datetime('now', 'Format', 'yyyyMMdd''T''HHmmss'));
    if ~isfolder(L.out_root), mkdir(L.out_root); end
    f = fullfile(L.out_root, ['night6_batch_' stamp '.json']);
    fid = fopen(f, 'w', 'n', 'UTF-8');
    fwrite(fid, jsonencode(summary, 'PrettyPrint', true), 'char');
    fclose(fid);
    fprintf('[night6] done: %d/%d ok in %.0f s -> %s\n', summary.n_ok, n, summary.wall_s, f);
end

function [s, w] = one(folder, meta, args)
    t = tic;
    try
        R = night6_run_recording(folder, 'MetaFile', meta, args{:});
        bad = cellfun(@(r) ~any(strcmp(r.status, {'complete', 'dry_run'})), R);
        if any(bad), s = 'failed (see night6_record.json)'; else, s = 'ok'; end
    catch ME
        s = sprintf('error: %s: %s', ME.identifier, ME.message);
        fprintf(2, '[night6] %s: %s\n', folder, s);
    end
    w = toc(t);
end

function preflight_recovery(folders, starts)
% Fail the whole batch early, by name, rather than once per recording after loading its
% signal: a stim_recovery epoch needs the declared starts, and the file must read.
    if ~isempty(starts)
        night6_recovery_start(starts);   % refuses an unreadable or malformed file
        return
    end
    need = {};
    for i = 1:numel(folders)
        files = dir(fullfile(folders{i}, 'e*_masks.mat'));
        for k = 1:numel(files)
            M = load(fullfile(files(k).folder, files(k).name), 'provenance_json');
            if ~isfield(M, 'provenance_json'), continue, end
            prov = jsondecode(char(M.provenance_json));
            if isfield(prov, 'extra') && isfield(prov.extra, 'condition') ...
                    && strcmp(prov.extra.condition, 'stim_recovery')
                need{end + 1} = fullfile(folders{i}, files(k).name); %#ok<AGROW>
            end
        end
    end
    if ~isempty(need)
        error('night6:recoveryStarts', ['%d stim_recovery epoch(s) listed and no ' ...
              'recovery_starts declared (RULING 2026-10-08 (k) 2), e.g. %s'], ...
              numel(need), need{1});
    end
end

function p = resolve(root, rel)
    rel = char(rel);
    if ~isempty(regexp(rel, '^([A-Za-z]:[\\/]|[\\/])', 'once'))
        p = rel;                        % absolute: a local list, never shared
    else
        parts = strsplit(rel, '/');
        p = fullfile(char(root), parts{:});
    end
end
