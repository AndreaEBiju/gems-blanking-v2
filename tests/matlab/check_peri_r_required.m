function check_peri_r_required(caseFile, outFile)
% CHECK_PERI_R_REQUIRED  Harness for tests/test_night6_wrapper.py: review fix 2 (2026-10-09).
%
% For each case of the JSON case file, loads the mask file (written by the real
% emit.handoff.write_mask_file, then possibly reduced to a pre-(k) 1 file) and plans its
% epoch with night6_prepare_epoch, no beats. Reports plan.periR's train_state and n_spans,
% 'not_read' when the spike consumer reads nothing, or the error identifier.
    C = jsondecode(fileread(caseFile));
    cases = C.cases;
    if ~iscell(cases), cases = num2cell(cases); end
    out = struct('name', {}, 'error', {}, 'message', {}, 'train_state', {}, 'n_spans', {});
    for k = 1:numel(cases)
        K = cases{k};
        r = struct('name', K.name, 'error', '', 'message', '', 'train_state', '', 'n_spans', -1);
        try
            M = load(K.mask_file);
            meta = jsondecode(K.meta_json);
            plan = night6_prepare_epoch(M, K.labels, meta.channels, K.n_file, K.fs, 'uV', [], ...
                                        'baseline');
            if isempty(plan.periR)
                r.train_state = 'not_read';
            else
                r.train_state = plan.periR.train_state;
                r.n_spans = plan.periR.n_spans;
            end
        catch ME
            r.error = ME.identifier;
            r.message = ME.message;
        end
        out(end + 1) = r; %#ok<AGROW>
    end
    fid = fopen(outFile, 'w', 'n', 'UTF-8');
    fwrite(fid, jsonencode(struct('cases', {num2cell(out)})), 'char');
    fclose(fid);
end
