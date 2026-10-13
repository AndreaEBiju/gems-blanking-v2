function n = fwrite(fid, data, varargin)
% Test stand-in (check_write_atomic): with N6_FAULT=short nothing reaches the file and 0 is
% returned, as on a full disk; otherwise the builtin.
    if strcmp(getenv('N6_FAULT'), 'short')
        n = 0;
    else
        n = builtin('fwrite', fid, data, varargin{:});
    end
end
