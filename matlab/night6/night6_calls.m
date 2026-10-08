function C = night6_calls()
% NIGHT6_CALLS  Which consumers each of Andrea's calls produces. Declared once.
%
% Invariant 32: the gate (run a call iff it produces a consumer that is wanted) and
% the recording (attribute exactly the wanted consumers to the run) are both derived
% from this table, never from hand-written flags. hrv and breathing are the two
% outputs of ONE HR_BR call - the pair that silently computed nothing for 105 points
% in T when it was gated on hrv alone (invariant 28).
%
% Order is the run order. velocity has no call: task 18 is out of this build (R5).
    C = struct( ...
        'name', {'detectSortNerveSpikesECAP', 'HR_BR_HRVAnalysis_beats', ...
                 'slowWaveAnalysis_new', 'extract_mmc'}, ...
        'consumers', {{'spikes'}, {'hrv', 'breathing'}, {'slow_wave'}, {'mmc'}});
end
