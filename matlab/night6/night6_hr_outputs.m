function U = night6_hr_outputs(consumers)
% NIGHT6_HR_OUTPUTS  Which outputs of one HR_BR_HRVAnalysis_beats call are read. ONE site.
%
%   U = night6_hr_outputs({'hrv'})  or  night6_hr_outputs({'breathing'})
%
% RULING 2026-10-09 (i) 4 (ruling 2026-10-08 (h) 8): hrv and breathing run as TWO calls,
% always - HRV and HR come from the hrv-masked call, breathing from the breathing-masked
% call. Each call still writes every variable (<label>_HRBR.mat, <label>_HRVMeasures.mat);
% the record names the ones this run is read for and the ones that are a byproduct of
% another consumer's mask and are NOT read from it. A run serving anything but exactly one
% of the two is refused by name ('night6:hrOutputs').
    consumers = cellstr(consumers);
    hr = {'HRBR: heartBeatSeries, heartlocs, heartRateSeries, avgHeartRate, heartCountSeries, avgHeartCount, heartCountValidSec, heartCountRateSeries, avgHeartCountRate, RR_implausibleMask', ...
          'HRVMeasures: every variable (hrv, rmssd, pnn5, sd1, sd2, sampEn, appxEn, RR_*, *_series, dfa_*)'};
    br = {'HRBR: breathRateSeries, avgBreathRate, br_locs_true, br_implausibleFraction'};
    if isequal(consumers, {'hrv'})
        U = struct('consumer', 'hrv', 'read', {hr}, 'byproduct_not_read', {br});
    elseif isequal(consumers, {'breathing'})
        U = struct('consumer', 'breathing', 'read', {br}, 'byproduct_not_read', {hr});
    else
        error('night6:hrOutputs', ['an HR_BR run serves exactly one of hrv / breathing ' ...
              '(RULING 2026-10-09 (i) 4); this one serves [%s]'], strjoin(consumers, ' '));
    end
end
