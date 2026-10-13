function check_write_atomic(resultFile, writerDir, faultDir)
% CHECK_WRITE_ATOMIC  night6_write_text_atomic under simulated write failures.
%
% WRITERDIR holds the night6_write_text_atomic.m under test (the real one, or a mutant).
% FAULTDIR holds fwrite.m / fclose.m stand-ins; they are copied into a private/ folder next
% to a copy of the writer, so they shadow the builtins for the writer only. getenv
% N6_FAULT picks the failure:
%   none     - nothing fails
%   short    - fwrite writes nothing and returns 0 (disk full before the first byte)
%   fclose   - fwrite succeeds, fclose's flush fails: the .tmp is left empty, status -1
%   truncate - as fclose, but fclose reports 0 (a failure nothing reports)
%   locked   - nothing fails while writing, but the record is held open by another
%              handle, so the rename onto it fails (Windows refuses to replace an open file)
% Each case starts from an existing record "OLD" and writes new text over it. The result
% file maps each case to {error_id, final_text, tmp_left}.
    out = struct();
    newText = sprintf('{\r\n  "new": "\x00B5V caf\x00E9"\r\n}');   % CRLF and non-ASCII
    want = unicode2native(strrep(newText, sprintf('\r\n'), newline), 'UTF-8');
    for mode = {'none', 'short', 'fclose', 'truncate', 'locked'}
        d = tempname;
        mkdir(fullfile(d, 'private'));
        copyfile(fullfile(writerDir, 'night6_write_text_atomic.m'), d);
        copyfile(fullfile(faultDir, 'fwrite.m'), fullfile(d, 'private'));
        copyfile(fullfile(faultDir, 'fclose.m'), fullfile(d, 'private'));
        addpath(d);
        assert(strcmp(fileparts(which('night6_write_text_atomic')), d), ...
               'the writer under test is not the one called');
        f = fullfile(d, 'night6_record.json');
        fid = builtin('fopen', f, 'w');
        builtin('fwrite', fid, 'OLD', 'char');
        builtin('fclose', fid);
        setenv('N6_FAULT', mode{1});
        held = -1;
        if strcmp(mode{1}, 'locked')
            held = builtin('fopen', f, 'r');   % another reader has the record open
        end
        r = struct('error_id', '', 'final_text', '', 'tmp_left', false, 'new_bytes_exact', false);
        try
            night6_write_text_atomic(f, newText);
        catch e
            r.error_id = e.identifier;
            r.message = e.message;
        end
        setenv('N6_FAULT', 'none');
        if held >= 0
            builtin('fclose', held);
        end
        fid = builtin('fopen', f, 'r');
        got = fread(fid, Inf, '*uint8');
        builtin('fclose', fid);
        r.final_text = char(got(:)');
        r.new_bytes_exact = isequal(got(:), want(:));
        r.tmp_left = isfile([f '.tmp']);
        out.(mode{1}) = r;
        rmpath(d);
        rmdir(d, 's');
    end
    fid = builtin('fopen', resultFile, 'w');
    builtin('fwrite', fid, jsonencode(out), 'char');
    builtin('fclose', fid);
end
