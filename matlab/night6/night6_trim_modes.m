function modes = night6_trim_modes()
% NIGHT6_TRIM_MODES  The two declared recovery trim semantics (RULING 2026-10-08 (k) 2).
%
%   modes = night6_trim_modes()
%
% The one list (invariant 33); gems_blanking_v2.extent.recovery_start.TRIM_MODES is the
% same two names, and a test holds them equal. A Night 6 batch must declare one
% (recovery_trim_mode, no default); a missing or unknown mode is refused by name.
%
%   mask_to_own_start                 each analysis's input is NaN up to its OWN start
%                                     (stim-off + electrical + own settling)
%   mask_to_electrical_drop_outputs   every analysis's input is NaN up to stim-off +
%                                     electrical settling, one point for all; outputs
%                                     stamped before the analysis's own start are then
%                                     dropped or flagged (night6_recovery_trim_outputs)
    modes = {'mask_to_own_start', 'mask_to_electrical_drop_outputs'};
end
