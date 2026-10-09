function [modes, withdrawn] = night6_trim_modes()
% NIGHT6_TRIM_MODES  The declared recovery trim mode, and the withdrawn ones.
%
%   [modes, withdrawn] = night6_trim_modes()
%
% The one list (invariant 33); gems_blanking_v2.extent.recovery_start.TRIM_MODES and
% WITHDRAWN_TRIM_MODES are the same names, and a test holds them equal. A Night 6 batch
% must still declare its mode (recovery_trim_mode, no default); a missing, unknown or
% withdrawn mode is refused by name (night6_check_trim_mode).
%
%   mask_to_electrical_drop_outputs   mode (B), RULING 2026-10-09 item 6: every analysis's
%                                     input is NaN only through stim-off + electrical
%                                     settling, one point for all; each output VARIABLE is
%                                     then cut at its own cut point by its class -
%                                     valid_only (i) at the electrical settling + its own
%                                     input's settling, filled_or_filtered (ii) at the
%                                     electrical settling + its full reach
%                                     (night6_recovery_trim_outputs)
%
% Withdrawn (refused by name, with the ruling):
%   mask_to_own_start                 mode (A): each analysis's input NaN up to its own start
    modes = {'mask_to_electrical_drop_outputs'};
    withdrawn = {'mask_to_own_start'};
end
