function night6_check(caseFile, outFile)
% NIGHT6_CHECK  Harness for tests/test_night6_wrapper.py (gems-blanking-v2, tests/matlab).
%
% Reads a JSON case file written by the Python test and, in ONE MATLAB process:
%   * inverts every Python-made token with night6_signal_from_token and forwards every
%     signal name with night6_token_from_signal (refusals reported as the error id);
%   * rounds every value with night6_round_half_even;
%   * runs night6_run_recording with DryRun on each synthetic mask folder (the real
%     loading, slicing, masking and skip logic; Andrea's functions are not called).
% Writes what it measured as JSON; the records land under the case's out_root.
    C = jsondecode(fileread(caseFile));
    out = struct();
    out.from_token = cellfun(@(t) attempt(@night6_signal_from_token, t), C.tokens, ...
                             'UniformOutput', false);
    out.to_token = cellfun(@(s) attempt(@night6_token_from_signal, s), C.signals, ...
                           'UniformOutput', false);
    out.rounded = night6_round_half_even(C.round_in(:)');
    folders = cellstr(C.mask_folders);
    out.errors = cell(1, numel(folders));
    for k = 1:numel(folders)
        try
            night6_run_recording(folders{k}, 'GemsRoot', C.gems_root, 'Units', C.units, ...
                'OutRoot', C.out_root, 'DryRun', true, 'CodeCommit', 'test');
            out.errors{k} = '';
        catch ME
            out.errors{k} = sprintf('%s: %s', ME.identifier, ME.message);
        end
    end
    % Resume: mark one epoch complete (right hash), one complete for ANOTHER mask file
    % (wrong hash), leave a stale file in both, and run that mask folder again.
    if isfield(C, 'resume')
        Rz = C.resume;
        for tag = {Rz.keep_tag, Rz.stale_tag}
            d = fullfile(Rz.out_dir, tag{1});
            f = fullfile(d, 'night6_record.json');
            txt = strrep(fileread(f), '"status": "dry_run"', '"status": "complete"');
            if strcmp(tag{1}, Rz.stale_tag)
                txt = regexprep(txt, '"mask_file_sha256": "[0-9a-f]+"', ...
                                '"mask_file_sha256": "deadbeef"');
            end
            fid = fopen(f, 'w', 'n', 'UTF-8'); fwrite(fid, txt, 'char'); fclose(fid);
            fid = fopen(fullfile(d, 'stale_marker.txt'), 'w'); fwrite(fid, 'x'); fclose(fid);
        end
        try
            night6_run_recording(Rz.mask_folder, 'GemsRoot', C.gems_root, 'Units', C.units, ...
                'OutRoot', C.out_root, 'DryRun', true, 'CodeCommit', 'test');
            out.resume_error = '';
        catch ME
            out.resume_error = sprintf('%s: %s', ME.identifier, ME.message);
        end
    end
    fid = fopen(outFile, 'w', 'n', 'UTF-8');
    fwrite(fid, jsonencode(out), 'char');
    fclose(fid);
end

function r = attempt(f, x)
    try
        r = f(x);
    catch ME
        r = ['!' ME.identifier];
    end
end
