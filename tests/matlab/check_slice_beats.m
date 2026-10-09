function check_slice_beats(caseFile, outFile)
% CHECK_SLICE_BEATS  Harness for tests/test_night6_wrapper.py: the beat train's origin.
%
% For each case of the JSON case file, builds the plan of one epoch with
% night6_prepare_epoch (a mask file reduced to its epoch fields: no masks, so only the
% beats are planned) and reports the epoch's beats - heartlocs (epoch rows), gapAfter,
% blankSpans - or the error identifier. The beats argument is built exactly as
% night6_run_recording builds it: struct(data = load(train file), record = jsondecode of
% the mask provenance's extra.beats_file, sha256 = night6_sha256_file(train file)).
    C = jsondecode(fileread(caseFile));
    cases = C.cases;
    if ~iscell(cases), cases = num2cell(cases); end
    out = struct('name', {}, 'error', {}, 'heartlocs', {}, 'gapAfter', {}, ...
                 'blankSpans', {}, 'nWholeFile', {}, 'originSample0', {});
    for k = 1:numel(cases)
        K = cases{k};
        r = struct('name', K.name, 'error', '', 'heartlocs', [], 'gapAfter', [], ...
                   'blankSpans', [], 'nWholeFile', [], 'originSample0', []);
        try
            M = struct('fs', K.fs, 'nSamples', K.n, 'epochStart_s', K.start_s, ...
                       'epochStartSample0', K.i0, 'notcomputed_json', '{}');
            sha = night6_sha256_file(K.beats_file);
            if isfield(K, 'sha256_override'), sha = K.sha256_override; end
            beats = struct('data', load(K.beats_file), 'record', jsondecode(K.record_json), ...
                           'sha256', sha);
            meta = struct('label', 'ANT1', 'role', 'stomach');
            plan = night6_prepare_epoch(M, {'ANT1'}, meta, K.n_file, K.fs, 'uV', beats, ...
                                        K.condition);
            E = plan.beats;
            r.heartlocs = E.heartlocs(:)';
            r.gapAfter = double(E.gapAfter(:)');
            r.blankSpans = E.blankSpans;
            r.nWholeFile = E.nWholeFile;
            r.originSample0 = E.originSample0;
        catch ME
            r.error = ME.identifier;
        end
        out(end + 1) = r; %#ok<AGROW>
    end
    fid = fopen(outFile, 'w', 'n', 'UTF-8');
    fwrite(fid, jsonencode(struct('cases', {num2cell(out)})), 'char');
    fclose(fid);
end
