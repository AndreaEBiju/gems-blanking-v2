function D = step1_bandpass(D, P, plotMode) %#ok<INUSD>
% STEP1_BANDPASS  TEST MUTANT (gems-blanking-v2 tests/matlab/mutants), NOT Andrea's step.
%
% A step 1 with the bug invariant 1 forbids: masked samples are zero-filled, filtered,
% and NOT restored to NaN, so D.filtered carries signal-looking values where the input
% was masked. Written for night6_check's mutant case only: process_dataset_v2 must refuse
% it by name ('process_dataset_v2:nanFilled'). It is put on the path only inside that
% case and removed straight after. Nothing of processing_new is copied here.
    ch = D.neuralChannels;
    [b, a] = butter(P.filterOrder, [P.bandpassLow P.bandpassHigh] / (D.fs / 2), 'bandpass');
    D.filtered = zeros(size(D.y, 1), numel(ch));
    for k = 1:numel(ch)
        x = D.y(:, ch(k));
        x(isnan(x)) = 0;                 % the bug: zero-fill ...
        D.filtered(:, k) = filtfilt(b, a, x);   % ... and never put the NaN back
    end
    D.bandInfo = struct('low', P.bandpassLow, 'high', P.bandpassHigh, 'mutant', true);
end
