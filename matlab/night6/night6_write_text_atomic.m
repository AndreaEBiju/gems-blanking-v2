function night6_write_text_atomic(f, txt)
%NIGHT6_WRITE_TEXT_ATOMIC  Write TXT to F as UTF-8 with \n line endings, atomically and checked.
%
%   The text goes to F.tmp first; F is replaced only after the temporary file is proven
%   to hold exactly TXT: fopen succeeded, fwrite wrote every byte, fclose returned 0, and
%   the file read back is byte-for-byte the encoded text. Any failure deletes the
%   temporary file, leaves F as it was, and raises night6:writeFailed naming F.
%   (Cross-platform rules 10, 11. Found 2026-10-12: on a full disk the unchecked
%   fwrite/fclose left a 0-byte .tmp that was then renamed over a night6_record.json.)
%
%   The one site that writes a Night 6 text record: night6_run_recording's write_json
%   and night6_batch's summary both call it.
    txt = strrep(char(txt), sprintf('\r\n'), newline);
    want = unicode2native(txt, 'UTF-8');
    tmp = [f '.tmp'];
    fid = fopen(tmp, 'w');                      % bytes, written as encoded above
    if fid < 0
        error('night6:writeFailed', 'cannot open %s for writing; %s left unchanged', tmp, f);
    end
    n = fwrite(fid, want, 'uint8');
    st = fclose(fid);
    if n ~= numel(want) || st ~= 0
        drop(tmp);
        error('night6:writeFailed', ['writing %s: %d of %d bytes written, fclose status %d ', ...
              '(disk full?); %s left unchanged'], tmp, n, numel(want), st, f);
    end
    fid = fopen(tmp, 'r');
    if fid < 0
        drop(tmp);
        error('night6:writeFailed', 'cannot read back %s; %s left unchanged', tmp, f);
    end
    got = fread(fid, Inf, '*uint8');
    fclose(fid);
    if ~isequal(got(:), want(:))
        drop(tmp);
        error('night6:writeFailed', ['%s reads back as %d bytes, not the %d written; ', ...
              '%s left unchanged'], tmp, numel(got), numel(want), f);
    end
    [ok, msg] = movefile(tmp, f, 'f');
    if ~ok
        drop(tmp);
        error('night6:writeFailed', 'cannot replace %s with %s: %s', f, tmp, msg);
    end
end

function drop(tmp)
    if isfile(tmp)
        delete(tmp);
    end
end
