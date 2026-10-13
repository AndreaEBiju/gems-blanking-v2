function st = fclose(fid)
% Test stand-in (check_write_atomic). For a file open for writing, N6_FAULT=fclose loses
% everything written (the buffered bytes never reach the disk) and returns -1, as fclose
% does when its flush fails on a full disk; N6_FAULT=truncate does the same but returns 0.
% Anything else: the builtin.
    mode = getenv('N6_FAULT');
    [name, perm] = builtin('fopen', fid);
    if any(strcmp(mode, {'fclose', 'truncate'})) && startsWith(perm, 'w')
        builtin('fclose', fid);
        g = builtin('fopen', name, 'w');   % truncate to 0 bytes
        builtin('fclose', g);
        st = -1 * strcmp(mode, 'fclose');
    else
        st = builtin('fclose', fid);
    end
end
