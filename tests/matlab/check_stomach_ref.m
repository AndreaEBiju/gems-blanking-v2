function check_stomach_ref(caseDir, outFile, outMat)
% CHECK_STOMACH_REF  Night 6's stomach_ref, from the mask provenance's recipe (Andrea,
% 2026-10-12), and the recipe gate (signals of running consumers only); results for
% tests/test_night6_stomach_ref.py.
%
% caseDir holds inputs.mat (Y, Ynan: the recording in its units; labels; fs; n; meta_json)
% and cases/*.mat, mask files written by emit.handoff.write_mask_file (or edited copies of
% one). Each case is planned by night6_prepare_epoch with a synthetic beat train on
% stomach_ref; the result per case is {error_id, message, statuses, recipes}. For the case
% 'ok' the stomach_ref recipe is applied to Y and Ynan (all rows, file units) and the hrv
% run's consumer input is formed; they are saved to outMat for Python to compare.
    I = load(fullfile(caseDir, 'inputs.mat'));
    labels = cellstr(I.labels);
    channels = jsondecode(I.meta_json);
    fs = double(I.fs);
    n = double(I.n);
    hl = (1000:4000:n - 1000)';
    sha = repmat('a', 1, 64);
    beats = struct('data', struct('fs', fs, 'heartlocs', hl, 'beatChannel', 'stomach_ref'), ...
                   'record', struct('origin_sample0', 0, 'sha256', sha, 'n_in_epoch', numel(hl)), ...
                   'sha256', sha);
    R = struct();
    files = dir(fullfile(caseDir, 'cases', '*.mat'));
    for k = 1:numel(files)
        name = erase(files(k).name, '.mat');
        M = load(fullfile(files(k).folder, files(k).name));
        r = struct('error_id', '', 'message', '', 'statuses', struct(), 'recipes', {{}}, ...
                   'stomach_cols', []);
        try
            plan = night6_prepare_epoch(M, labels, channels, n, fs, 'uV', beats, 'baseline');
            for c = {'hrv', 'breathing', 'mmc'}
                r.statuses.(c{1}) = plan.consumers.(c{1}).status;
            end
            r.recipes = fieldnames(plan.recipes)';
            if isfield(plan.recipes, 'stomach_ref')
                r.stomach_cols = plan.recipes.stomach_ref.cols;
            end
            if strcmp(name, 'ok')
                rows = (1:n)';
                x = night6_apply_recipe(plan.recipes.stomach_ref, I.Y, rows); %#ok<NASGU>
                x_nan = night6_apply_recipe(plan.recipes.stomach_ref, I.Ynan, rows); %#ok<NASGU>
                run = plan.runs(strcmp(cellfun(@(c) c{1}, {plan.runs.consumers}, ...
                                               'UniformOutput', false), 'hrv'));
                X_hrv = night6_consumer_input(plan, I.Y, run); %#ok<NASGU>
                save(outMat, 'x', 'x_nan', 'X_hrv');
            end
        catch e
            r.error_id = e.identifier;
            r.message = e.message;
        end
        R.(name) = r;
    end
    % Wide stomach arrays (review 2026-10-12): 7 contacts are built, 8 are refused by name.
    W = load(fullfile(caseDir, 'inputs_wide.mat'));
    for nc = [7 8]
        tag = sprintf('wide%d', nc);
        r = struct('error_id', '', 'message', '');
        try
            chw = jsondecode(W.(['meta' num2str(nc)]));
            if isstruct(chw), chw = num2cell(chw(:)'); end
            rec = night6_signal_recipe('stomach_ref', cellstr(W.(['labels' num2str(nc)])), ...
                                       chw, jsondecode(W.(['recipe' num2str(nc)])));
            xw = night6_apply_recipe(rec, W.(['Y' num2str(nc)]), (1:size(W.(['Y' num2str(nc)]), 1))');
            S = struct(['x' num2str(nc)], xw);
            if isfile(outMat)
                save(outMat, '-struct', 'S', '-append');
            else
                save(outMat, '-struct', 'S');
            end
        catch e
            r.error_id = e.identifier;
            r.message = e.message;
        end
        R.(tag) = r;
    end
    fid = fopen(outFile, 'w');
    fwrite(fid, jsonencode(R), 'char');
    fclose(fid);
end
