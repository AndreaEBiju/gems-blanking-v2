function W = night6_sw_windows(F, startS)
% NIGHT6_SW_WINDOWS  Her slow-wave rate windows: valid fraction, fit, and the per-window rule.
%
%   W = night6_sw_windows(F, startS)
%
%   F       one kept channel's output file, loaded (slowWaveRateTime, slowWaveRateSeries,
%           invalidMask, rateWinSec, fs; slow_wave_span_rule.zone_runs_called when present)
%   startS  epoch seconds of the slow-wave start (the stim-recovery trim's cut; 0 otherwise)
%
% ONE site for the window definitions Night 6 records (RULING 2026-10-12 step 3, RULING
% 2026-10-13 slow-wave release). night6_recovery_trim_outputs writes the same valid fraction
% as slowWaveRateSeries_validFraction (equality tested, tests/test_night6_slow_wave.py):
%   valid rows      ~invalidMask & ~(the span rule's < 10 s zones), at the rate she ran at
%   valid_fraction  for the window centred on her slowWaveRateTime(i): centre row
%                   round(t fs) + 1, half-width round(rateWinSec fs / 2), clipped to 1..N
%   fits            her own edge rule: the centred window fits inside the epoch,
%                   rateWinSec/2 <= t <= (N - 1)/fs - rateWinSec/2 (slowWaveAnalysis_new.m)
%   counted         fits and t >= startS (her rate is not computed elsewhere)
%   valid           counted and valid_fraction >= 0.5 (the per-window 50% rule)
% W.n_windows, n_valid_windows, n_rate_windows (counted with a finite rate), and W.status:
%   'ran'              some counted window has a rate;
%   'no valid window'  no counted window is valid (RULING 2026-10-13: no rate, never held);
%   'ran: no rate'     valid windows, none with a rate (too few peaks).
    fs = double(F.fs);
    inv = logical(F.invalidMask(:));
    N = numel(inv);
    valid = ~inv;
    if isfield(F, 'slow_wave_span_rule') && isfield(F.slow_wave_span_rule, 'rule') ...
            && isfield(F.slow_wave_span_rule.rule, 'exclusionFromCleanTime') ...
            && F.slow_wave_span_rule.rule.exclusionFromCleanTime   % as the trim (span_rule_zone)
        rr = reshape(double(F.slow_wave_span_rule.zone_runs_called), [], 2);
        for k = 1:size(rr, 1)
            valid(rr(k, 1):rr(k, 2)) = false;
        end
    end
    t = double(F.slowWaveRateTime(:));
    win = double(F.rateWinSec);
    c = round(t * fs) + 1;
    h = round(win * fs / 2);
    lo = max(1, c - h);
    hi = min(N, c + h);
    cs = [0; cumsum(valid)];
    fr = NaN(numel(t), 1);
    ok = hi >= lo;
    fr(ok) = (cs(hi(ok) + 1) - cs(lo(ok))) ./ (hi(ok) - lo(ok) + 1);
    fits = t >= win / 2 & t <= (N - 1) / fs - win / 2;
    counted = fits & t >= startS - 1e-9;
    rate = double(F.slowWaveRateSeries(:));
    W = struct('valid_fraction', fr, 'fits', fits, 'counted', counted, ...
               'n_windows', nnz(counted), 'n_valid_windows', nnz(counted & fr >= 0.5), ...
               'n_rate_windows', nnz(counted & isfinite(rate)));
    if W.n_rate_windows > 0
        W.status = 'ran';
    elseif W.n_valid_windows == 0
        W.status = 'no valid window';
    else
        W.status = 'ran: no rate';
    end
end
