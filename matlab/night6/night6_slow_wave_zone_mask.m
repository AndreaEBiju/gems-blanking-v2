function z = night6_slow_wave_zone_mask(epochRows, spans, fsEpoch, sec)
% NIGHT6_SLOW_WAVE_ZONE_MASK  Rows less than sec from a masked span, or inside one (RULING 2026-10-12 (b)).
%
%   z = night6_slow_wave_zone_mask(epochRows, spans, fsEpoch, sec)
%
%   epochRows  the EPOCH row (1-based, at fsEpoch) of each row of the series to mask - for her
%              decimated series, row j is epoch row (j - 1) R + 1 (night6_decimate_masked)
%   spans      [nSeg x 2] 1-based inclusive epoch rows of the masked spans (may be empty)
%   sec        the exclusion distance (spanRule.exclusionSec)
%
% z(j) is true when epoch row r = epochRows(j) has distance < sec to a span, with the same
% distance as night6_slow_wave_span_rule: (a - r) / fs before span [a, b], (r - b) / fs after
% it, 0 inside. These rows are left out of the clean time her rate pools (rule (b)).
    r = double(epochRows(:));
    z = false(numel(r), 1);
    if isempty(spans), return, end
    spans = double(spans);
    for k = 1:size(spans, 1)
        a = spans(k, 1); b = spans(k, 2);
        z = z | ((r >= a & r <= b) | (r < a & (a - r) / fsEpoch < sec) ...
                 | (r > b & (r - b) / fsEpoch < sec));
    end
end
