function night6_write_text_atomic(f, txt)
% MUTANT (revert check for tests/test_night6_write_atomic.py): the record writer as it was
% before 2026-10-12 - nothing checks the write before the rename.
    tmp = [f '.tmp'];
    fid = fopen(tmp, 'w', 'n', 'UTF-8');
    fwrite(fid, strrep(txt, sprintf('\r\n'), newline), 'char');
    fclose(fid);
    movefile(tmp, f, 'f');
end
