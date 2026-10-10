function u = night6_hold_unit(consumer, signal)
% NIGHT6_HOLD_UNIT  The unit one mask is held in - the ONE site on the MATLAB side.
%
%   u = night6_hold_unit(consumer, signal)
%
% Andrea, revised 2026-10-10: holds are per consumer, and for spikes per cuff - a distrusted
% or held cuff never holds the other cuff. The unit is the consumer, except spikes:
% 'spikes_<cuff>' for a signal '<cuff>_T' or '<cuff>_V<k>'. Python twin: emit.qc.hold_unit.
% A spike signal with no cuff is refused (night6:holdUnit): its unit is never guessed.
    consumer = char(consumer);
    if ~strcmp(consumer, 'spikes')
        u = consumer;
        return
    end
    t = regexp(char(signal), '^([A-Za-z0-9]+)_(V\d+|T)$', 'tokens', 'once');
    if isempty(t)
        error('night6:holdUnit', 'spike signal %s names no cuff: its hold unit is unknown', ...
              char(signal));
    end
    u = ['spikes_' t{1}];
end
