function C = night6_calls()
% NIGHT6_CALLS  Which consumers each call produces. Declared once.
%
% Invariant 32: the gate (run a call iff it produces a consumer that is wanted) and
% the recording (attribute exactly the wanted consumers to the run) are both derived
% from this table, never from hand-written flags. hrv and breathing are the two
% outputs of ONE HR_BR call - the pair that silently computed nothing for 105 points
% in T when it was gated on hrv alone (invariant 28).
%
% spikes: process_dataset_v2 (this folder), which calls Andrea's process_dataset steps
% unchanged (Andrea, 2026-10-09; it replaced detectSortNerveSpikesECAP). Every other
% call is Andrea's own function in processing_new, called unchanged.
%
% Order is the run order. velocity has no call: task 18 is out of this build (R5).
    C = struct( ...
        'name', {'process_dataset_v2', 'HR_BR_HRVAnalysis_beats', ...
                 'slowWaveAnalysis_new', 'extract_mmc'}, ...
        'consumers', {{'spikes'}, {'hrv', 'breathing'}, {'slow_wave'}, {'mmc'}});
end
