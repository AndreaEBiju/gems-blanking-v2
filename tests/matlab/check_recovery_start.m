function check_recovery_start(caseFile, outFile)
% CHECK_RECOVERY_START  Harness for tests/test_night6_recovery_start.py (RULING 2026-10-08 (k) 2).
%
%   plans    each case plans ONE epoch from a mask file written by the real
%            emit.handoff.write_mask_file, twice: without 'Recovery' (the untrimmed
%            baseline) and with the case's starts file. For both it reports, per run and
%            per input column, the NaN rows of night6_consumer_input (1-based inclusive
%            runs), plus plan.recoveryStart - or the error identifier and message.
%   readers  night6_recovery_start on each file: 'ok' or the error identifier.
%   run      night6_run_recording (DryRun) on a store mask folder: without RecoveryStarts,
%            with one, and the resume rule (a complete record made with another starts
%            file reruns; one made with this file is skipped).
    C = jsondecode(fileread(caseFile));
    out = struct();
    out.plans = cellfun(@plan_case, as_cells(C.plans), 'UniformOutput', false);
    out.readers = cellfun(@reader_case, as_cells(C.readers), 'UniformOutput', false);
    if isfield(C, 'run'), out.run = run_case(C.run); end
    fid = fopen(outFile, 'w', 'n', 'UTF-8');
    fwrite(fid, jsonencode(out), 'char');
    fclose(fid);
end

function r = plan_case(K)
    r = struct('name', K.name, 'error', '', 'message', '', 'base', {{}}, 'trim', {{}}, ...
               'record', '');
    try
        M = load(K.mask_file);
        meta = jsondecode(K.meta_json);
        beats = [];
        if isfield(K, 'beats_file') && ~isempty(K.beats_file)
            beats = struct('data', load(K.beats_file), 'record', jsondecode(K.record_json), ...
                           'sha256', night6_sha256_file(K.beats_file));
        end
        Y = ones(K.n_file, numel(K.labels));
        base = night6_prepare_epoch(M, K.labels, meta.channels, K.n_file, K.fs, 'uV', beats, ...
                                    K.condition);
        r.base = describe(base, Y);
        RS = night6_recovery_start(K.starts_file);
        plan = night6_prepare_epoch(M, K.labels, meta.channels, K.n_file, K.fs, 'uV', beats, ...
                                    K.condition, 'Recovery', struct('starts', RS, ...
                                                                    'session', K.session));
        r.trim = describe(plan, Y);
        r.record = jsonencode(plan.recoveryStart);
    catch ME
        r.error = ME.identifier;
        r.message = ME.message;
    end
end

function d = describe(plan, Y)
    d = {};
    for run = plan.runs
        X = night6_consumer_input(plan, Y, run);
        for j = 1:size(X, 2)
            b = isnan(X(:, j));
            e = diff([false; b; false]);
            d{end + 1} = struct('call', run.call, 'consumers', {run.consumers}, ...
                                'signal', run.signals{j}, 'mask_signal', run.maskSignal, ...
                                'keep', {run.keep}, ...
                                'nan_runs', [find(e == 1), find(e == -1) - 1]); %#ok<AGROW>
        end
    end
end

function r = reader_case(K)
    r = struct('name', K.name, 'error', '', 'message', '');
    try
        night6_recovery_start(K.file);
    catch ME
        r.error = ME.identifier;
        r.message = ME.message;
    end
end

function r = run_case(K)
    args = {'GemsRoot', K.gems_root, 'Units', 'uV', 'DryRun', true, 'CodeCommit', 'test'};
    r = struct();
    r.without = attempt(@() night6_run_recording(K.mask_folder, 'OutRoot', K.out_a, args{:}));
    R = night6_run_recording(K.mask_folder, 'OutRoot', K.out_b, 'RecoveryStarts', ...
                             K.starts_file, args{:});
    r.with = cellfun(@(x) jsonencode(x), R, 'UniformOutput', false);
    % resume: a COMPLETE record made with this starts file is skipped; with another, rerun
    recFile = fullfile(K.out_b, K.record_rel);
    Rc = jsondecode(fileread(recFile));
    Rc.status = 'complete';
    write(recFile, Rc);
    R2 = night6_run_recording(K.mask_folder, 'OutRoot', K.out_b, 'RecoveryStarts', ...
                              K.starts_file, args{:});
    r.resume_same = R2{1}.status;
    R3 = night6_run_recording(K.mask_folder, 'OutRoot', K.out_b, 'RecoveryStarts', ...
                              K.other_starts_file, args{:});
    r.resume_other = R3{1}.status;
end

function s = attempt(f)
    s = struct('error', '', 'message', '');
    try
        f();
    catch ME
        s.error = ME.identifier;
        s.message = ME.message;
    end
end

function write(f, R)
    fid = fopen(f, 'w', 'n', 'UTF-8');
    fwrite(fid, jsonencode(R), 'char');
    fclose(fid);
end

function c = as_cells(v)
    if isempty(v), c = {}; elseif isstruct(v), c = num2cell(v(:))'; else, c = v(:)'; end
end
