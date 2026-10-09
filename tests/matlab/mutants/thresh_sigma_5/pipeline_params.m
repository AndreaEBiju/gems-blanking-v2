function P = pipeline_params()
% PIPELINE_PARAMS  TEST MUTANT (gems-blanking-v2 tests/matlab/mutants), NOT Andrea's.
%
% Stands for a change to her default detection threshold (4.5 -> 5). Only the fields
% process_dataset_v2 reads before its first step are set: night6_v2_params must refuse
% it by name ('process_dataset_v2:params') before any step runs. Put on the path only
% inside night6_check's mutant case and removed straight after.
    P = struct('threshSigma', 5, 'bandpassLow', 300, 'bandpassHigh', 3000, 'filterOrder', 4);
end
