function mode = night6_check_trim_mode(mode)
% NIGHT6_CHECK_TRIM_MODE  Refuse a missing, unknown or withdrawn recovery trim mode, by name.
%
%   mode = night6_check_trim_mode(mode)
%
% 'night6:recoveryTrimMode' unless mode is one of night6_trim_modes(). There is no
% default: the batch list must declare recovery_trim_mode (RULING 2026-10-08 (k) 2). A
% withdrawn mode - (A) mask_to_own_start - is refused naming RULING 2026-10-09 item 6, so
% a stale batch list cannot run it.
    [modes, withdrawn] = night6_trim_modes();
    if isstring(mode), mode = char(mode); end
    if isempty(mode) || ~ischar(mode)
        error('night6:recoveryTrimMode', ['recovery_trim_mode is required and has no ' ...
              'default: declare one of %s'], strjoin(modes, ', '));
    end
    if any(strcmp(withdrawn, mode))
        error('night6:recoveryTrimMode', ['recovery_trim_mode ''%s'' is trim mode (A), ' ...
              'withdrawn by RULING 2026-10-09 item 6: Night 6 runs mode (B), %s (input ' ...
              'masked through the input mask point only, outputs trimmed per variable)'], ...
              mode, strjoin(modes, ', '));
    end
    if ~any(strcmp(modes, mode))
        error('night6:recoveryTrimMode', 'unknown recovery_trim_mode ''%s'': expected one of %s', ...
              mode, strjoin(modes, ', '));
    end
end
