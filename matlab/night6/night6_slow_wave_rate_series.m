function [rate, rateTime] = night6_slow_wave_rate_series(locs, invalidMask, cleanMask, fs, ...
                                                          rateWinSec, minStretchSec, minPeaksInStretch)
% NIGHT6_SLOW_WAVE_RATE_SERIES  Her slow-wave rate series, from a GIVEN peak list.
%
%   [rate, rateTime] = night6_slow_wave_rate_series(locs, invalidMask, cleanMask, fs, ...
%                                                   rateWinSec, minStretchSec, minPeaksInStretch)
%
% RULING 2026-10-12 (d): rate outputs use only the kept peaks. slowWaveAnalysis_new cannot be
% handed a peak list, so its rate loop (processing_new f93e250 slowWaveAnalysis_new.m:169-283:
% 1 s steps, centred 60 s window, NaN where the window does not fit or its centre is
% invalid, every clean run of the window pooled, >= minStretchSec pooled and
% >= minPeaksInStretch peaks) is reproduced here statement for statement, in her operation
% order. night6_call_slow_wave ASSERTS, on every call, that this function given her own
% peaks and her own clean mask (~invalidMask & ~edgeMask) returns her slowWaveRateSeries
% bit for bit ('night6:rateReplica') before it is used on the kept peaks - so a change in
% her code cannot silently leave this copy behind.
%
%   locs         peak rows at the rate she ran at (1-based)
%   invalidMask  her invalidMask (N x 1)
%   cleanMask    the rows counted as clean time (hers: ~invalidMask & ~edgeMask)
%   fs           the rate she ran at
%   rate         nRateT x 1, cpm (NaN = not computed);  rateTime  her slowWaveRateTime
    invalidMask = logical(invalidMask(:));
    cleanMask = logical(cleanMask(:));
    locs = double(locs(:));
    N = numel(invalidMask);
    t = (0:N-1)' / fs;
    rateStepSamp = round(fs);
    rateT_idx = 1 : rateStepSamp : N;
    rateT_idx = rateT_idx(rateT_idx >= 1 & rateT_idx <= N);
    nRateT = numel(rateT_idx);
    rateTime = t(rateT_idx);
    rate = NaN(nRateT, 1);
    minStretchSamp = round(minStretchSec * fs);
    sigDurSec = (N - 1) / fs;
    halfWinSamp = round(rateWinSec * fs / 2);
    halfWinSec = rateWinSec / 2;
    for ti = 1:nRateT
        ctrSamp = rateT_idx(ti);
        tc = rateTime(ti);
        if tc < halfWinSec || tc > sigDurSec - halfWinSec
            continue;
        end
        if invalidMask(ctrSamp)
            continue;
        end
        winStartSamp = max(1, ctrSamp - halfWinSamp);
        winEndSamp = min(N, ctrSamp + halfWinSamp);
        runs = contiguous_runs(cleanMask(winStartSamp:winEndSamp));
        pooledLen = 0;
        pooledPeaks = 0;
        for rr = 1:size(runs, 1)
            absStart = winStartSamp + runs(rr, 1) - 1;
            absEnd = min(N, winStartSamp + runs(rr, 2) - 1);
            pooledLen = pooledLen + (absEnd - absStart + 1);
            pooledPeaks = pooledPeaks + sum(locs >= absStart & locs <= absEnd);
        end
        if pooledLen < minStretchSamp
            continue;
        end
        if pooledPeaks >= minPeaksInStretch
            stretchDurSec = pooledLen / fs;
            rate(ti) = pooledPeaks / (stretchDurSec / 60);
        end
    end
end

function runs = contiguous_runs(v)
% her contiguousRuns (slowWaveAnalysis_new.m): [nRuns x 2] 1-based [start, end]
    if ~any(v)
        runs = zeros(0, 2); return;
    end
    d = diff([0; v(:); 0]);
    runs = [find(d == 1), find(d == -1) - 1];
end
