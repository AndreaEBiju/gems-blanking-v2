function [amp, troughRow] = night6_slow_wave_peak_to_trough(ys, detected, kept)
% NIGHT6_SLOW_WAVE_PEAK_TO_TROUGH  Peak-to-trough amplitude of each kept slow-wave peak.
%
%   [amp, troughRow] = night6_slow_wave_peak_to_trough(ys, detected, kept)
%
%   ys        her filtered trace, slowWaveTimeSeries, at the rate she ran at (NaN in spans)
%   detected  her DETECTED peak rows (before the span rule), at that rate, ascending
%   kept      the kept peak rows (a subset of detected), at that rate
%
% Andrea, 2026-10-12: the pre-span diagnostic's amplitude is peak-to-trough, because the
% signed peak value of a detrended trace can have a median <= 0. For kept peak c:
%   the PRECEDING TROUGH is the minimum of ys over the rows strictly between the previous
%   DETECTED peak (kept or dropped; row 0 when c is the first) and c, cut back to the rows
%   after the last NaN in that stretch - so a trough is never taken across a masked span;
%   amp = ys(c) - trough;   troughRow = the row of that minimum (the first, on ties).
% When that stretch is empty (c follows its previous peak or a NaN directly) amp and
% troughRow are NaN: no amplitude is made up (invariant 8).
    ys = double(ys(:));
    detected = double(detected(:));
    kept = double(kept(:));
    amp = NaN(numel(kept), 1);
    troughRow = NaN(numel(kept), 1);
    for i = 1:numel(kept)
        c = kept(i);
        p = detected(detected < c);
        if isempty(p), p = 0; else, p = p(end); end
        lo = p + 1;
        hi = c - 1;
        if hi < lo, continue, end
        last = find(isnan(ys(lo:hi)), 1, 'last');
        if ~isempty(last), lo = lo + last; end
        if hi < lo, continue, end
        [m, k] = min(ys(lo:hi));
        amp(i) = ys(c) - m;
        troughRow(i) = lo + k - 1;
    end
end
