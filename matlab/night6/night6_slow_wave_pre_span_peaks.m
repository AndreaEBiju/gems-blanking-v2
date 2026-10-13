function P = night6_slow_wave_pre_span_peaks(rows, amps, spans, fsEpoch, D, troughRows)
% NIGHT6_SLOW_WAVE_PRE_SPAN_PEAKS  DIAGNOSTIC: large kept peaks shortly BEFORE a masked span.
%
%   P = night6_slow_wave_pre_span_peaks(rows, amps, spans, fsEpoch, D)
%   P = night6_slow_wave_pre_span_peaks(..., troughRows)
%
%   rows     the KEPT peaks, epoch rows (1-based, ascending)
%   amps     their PEAK-TO-TROUGH amplitudes (night6_slow_wave_peak_to_trough; Andrea
%            2026-10-12: the signed peak value of a detrended trace can have a median <= 0),
%            the input's units (volts); NaN where no trough is defined - such a peak is left
%            out of the median and never listed
%   troughRows  optional: each peak's trough, epoch rows (recorded beside the peak)
%   spans    the run's masked spans, [nSeg x 2] 1-based inclusive epoch rows
%   D        night6_slow_wave_settings().preSpanDiagnostic: windowSec (25), ampFactor (2)
%
% Andrea, 2026-10-12: list every kept peak that lies within D.windowSec BEFORE a span's start
% (0 < (start - row) / fs <= windowSec, the next span after the peak) and whose amplitude is
% more than D.ampFactor x the median amplitude of the channel's kept peaks in this epoch.
% A DIAGNOSTIC ONLY: it reads the kept peaks and changes nothing - no peak, rate, average or
% fraction depends on it. The median is over the finite amplitudes; with none, or a median
% <= 0, the threshold is not defined: P.defined is false and nothing is listed.
%
% P.median, P.threshold, P.defined, P.n, and per listed peak (column vectors): row,
% t_epoch_s = (row - 1) / fsEpoch, amplitude (input units), amplitude_uV (x 1e6, volts in),
% dist_s (to the span start), span (its index), span_start_row.
    rows = double(rows(:));
    amps = double(amps(:));
    if nargin < 6 || isempty(troughRows), troughRows = NaN(size(rows)); end
    troughRows = double(troughRows(:));
    if numel(rows) ~= numel(amps)
        error('night6:preSpan', 'one amplitude per kept peak: %d rows, %d amplitudes', ...
              numel(rows), numel(amps));
    end
    for f = {'windowSec', 'ampFactor'}
        if ~isfield(D, f{1}) || ~isscalar(D.(f{1})) || ~(D.(f{1}) > 0)
            error('night6:preSpan', 'D.%s is required (> 0)', f{1});
        end
    end
    if isempty(spans), spans = zeros(0, 2); end
    spans = double(spans);
    fin = amps(isfinite(amps));
    med = NaN;
    if ~isempty(fin), med = median(fin); end
    P = struct('median', med, 'threshold', D.ampFactor * med, 'defined', med > 0, ...
               'window_s', D.windowSec, 'amp_factor', D.ampFactor, 'n', 0, ...
               'row', zeros(0, 1), 't_epoch_s', zeros(0, 1), 'amplitude', zeros(0, 1), ...
               'amplitude_uV', zeros(0, 1), 'dist_s', zeros(0, 1), 'span', zeros(0, 1), ...
               'span_start_row', zeros(0, 1), 'trough_t_epoch_s', zeros(0, 1), ...
               'amplitude_kind', 'peak-to-trough (night6_slow_wave_peak_to_trough)', ...
               'what', ['DIAGNOSTIC only (Andrea 2026-10-12): kept peaks within window_s before ' ...
                        'a masked span''s start, amplitude > amp_factor x the median kept-peak ' ...
                        'amplitude; no rate depends on it']);
    if ~P.defined || isempty(spans), return, end
    starts = sort(spans(:, 1));
    [~, order] = sort(spans(:, 1));
    for i = 1:numel(rows)
        k = find(starts > rows(i), 1);   % the next span after the peak
        if isempty(k), continue, end
        d = (starts(k) - rows(i)) / fsEpoch;
        if d > 0 && d <= D.windowSec && isfinite(amps(i)) && amps(i) > P.threshold
            P.row(end + 1, 1) = rows(i);
            P.t_epoch_s(end + 1, 1) = (rows(i) - 1) / fsEpoch;
            P.amplitude(end + 1, 1) = amps(i);
            P.amplitude_uV(end + 1, 1) = amps(i) * 1e6;
            P.dist_s(end + 1, 1) = d;
            P.span(end + 1, 1) = order(k);
            P.span_start_row(end + 1, 1) = starts(k);
            P.trough_t_epoch_s(end + 1, 1) = (troughRows(i) - 1) / fsEpoch;
        end
    end
    P.n = numel(P.row);
end
