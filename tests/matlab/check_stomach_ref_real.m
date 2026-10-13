function check_stomach_ref_real(caseFile, units, outFile, outMat)
% CHECK_STOMACH_REF_REAL  Night 6's stomach_ref on a real recording, from the recipe in a
% re-versioned mask file's provenance; for tests/test_night6_stomach_ref.py
% (GEMS_STOMACHREF_REAL names caseFile, written by stomach_ref_proof.py).
%
% Saves x (the recipe applied to the epoch's file rows, in the file's units) to outMat, and
% writes to outFile: whether every variable of the new mask file but provenance_json is
% identical (class, size, value) to the original's; the recipe's columns; and the plan of
% the new file with the recording's real beat train (statuses, recipes) plus, when hrv
% runs, whether its consumer input equals x in volts outside its mask.
    C = jsondecode(fileread(caseFile));
    root = C.gems_root;
    S = load(fullfile(root, C.source_path), 'signal', 'fs', 'chanlabels');
    meta = jsondecode(fileread(C.meta_file));
    channels = meta.channels;
    if isstruct(channels), channels = num2cell(channels(:)'); end
    Mo = load(C.old_mask);
    Mn = load(C.new_mask);
    R = struct();
    fo = sort(fieldnames(Mo));
    fn = sort(fieldnames(Mn));
    R.same_fields = isequal(fo, fn);
    diff = {};
    for k = 1:numel(fo)
        f = fo{k};
        if strcmp(f, 'provenance_json') || ~isfield(Mn, f), continue, end
        if ~strcmp(class(Mo.(f)), class(Mn.(f))) || ~isequal(size(Mo.(f)), size(Mn.(f))) ...
                || ~isequal(Mo.(f), Mn.(f))
            diff{end + 1} = f; %#ok<AGROW>
        end
    end
    R.differing = diff;
    P = jsondecode(char(Mn.provenance_json));
    r = night6_signal_recipe('stomach_ref', cellstr(S.chanlabels), channels, P.stomach_ref_recipe);
    R.cols = r.cols;
    labels = cellstr(S.chanlabels);
    R.labels = labels(r.cols);
    R.signal_class = class(S.signal);
    i0 = double(Mn.epochStartSample0);
    n = double(Mn.nSamples);
    x = night6_apply_recipe(r, S.signal, i0 + (1:n)');
    save(outMat, 'x');
    R.plan_error = '';
    try
        bf = fullfile(root, P.extra.beats_file.hrv_beats);
        T = struct('data', load(bf), 'record', P.extra.beats_file, ...
                   'sha256', night6_sha256_file(bf));
        plan = night6_prepare_epoch(Mn, S.chanlabels, meta.channels, size(S.signal, 1), ...
                                    S.fs, units, T, char(P.extra.condition));
        R.statuses = struct('hrv', plan.consumers.hrv.status, ...
                            'breathing', plan.consumers.breathing.status, ...
                            'mmc', plan.consumers.mmc.status);
        R.recipes = fieldnames(plan.recipes)';
        sel = strcmp(cellfun(@(c) c{1}, {plan.runs.consumers}, 'UniformOutput', false), 'hrv');
        if any(sel)
            X = night6_consumer_input(plan, S.signal, plan.runs(sel));
            m = ~isnan(X);
            R.hrv_input_equals_x_outside_mask = isequal(X(m), x(m) * plan.scaleToVolts);
            R.hrv_masked_rows = nnz(~m);
        end
    catch e
        R.plan_error = [e.identifier ': ' e.message];
    end
    fid = fopen(outFile, 'w');
    fwrite(fid, jsonencode(R), 'char');
    fclose(fid);
end
