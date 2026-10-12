function K = night6_slow_wave_span_rule(peaks, spans, fs, rule)
% NIGHT6_SLOW_WAVE_SPAN_RULE  Which slow-wave peaks the masked spans let through (RULING 2026-10-12).
%
%   K = night6_slow_wave_span_rule(peaks, spans, fs, rule)
%
%   peaks  her detected peaks, epoch rows (1-based), ascending - her list after her own
%          invalid / file-edge rejection (slowWavePeakLocs as night6_keep_slow_wave keeps it)
%   spans  the run's masked spans, [nSeg x 2] 1-based inclusive epoch rows (may be empty)
%   fs     the epoch's rate (Hz) - rows and spans are at this rate
%   rule   night6_slow_wave_settings().spanRule: exclusionSec, pairZoneSec, pairMinSec
%
% (b) a peak is dropped when its distance to a masked span is < exclusionSec (inside a span
%     the distance is 0). Distance to span [a, b] (rows): (a - p) / fs before it,
%     (p - b) / fs after it, 0 inside; a peak's distance is its minimum over the spans.
% (c) two CONSECUTIVE peaks of her list, (p(i), p(i+1)) with (p(i+1) - p(i)) / fs < pairMinSec,
%     are BOTH dropped when BOTH are within pairZoneSec (distance <= pairZoneSec) of a masked
%     span - the zone is the union over spans, either side. "Consecutive" is her list as
%     detected (before (b)): (b) and (c) are each judged on her list and a peak goes if
%     either says so. K.pairs_c_after_b says what (c) would drop if it were judged on the
%     peaks (b) keeps instead - the other reading of the wording, reported, never applied.
%
% K.kept           logical, one per peak: neither (b) nor (c)
% K.dropped_b      logical: (b)  (inside a span or < exclusionSec from one)
% K.dropped_c      logical: (c) and NOT (b)  (so the counts add up: kept + b + c = n)
% K.in_c_pair      logical: (c), whether or not (b) also holds
% K.inside         logical: inside a span (a subset of dropped_b)
% K.dist_s         distance (s) to the nearest span (Inf with no spans)
% K.in_zone        logical: dist_s <= pairZoneSec
% K.pairs_c        [nPair x 2] indices into peaks of the (c) pairs (her list)
% K.pairs_c_spans  [nPair x 2] for each (c) pair, the nearest span of each peak (index)
% K.pairs_c_after_b  [nPair x 2] the (c) pairs judged on (b)'s survivors instead (reported)
% K.n, n_kept, n_b, n_c  counts
    peaks = double(peaks(:));
    n = numel(peaks);
    if n > 1 && any(diff(peaks) <= 0)
        error('night6:spanRule', 'peaks must be strictly ascending epoch rows');
    end
    if isempty(spans), spans = zeros(0, 2); end
    spans = double(spans);
    if any(spans(:, 2) < spans(:, 1))
        error('night6:spanRule', 'a span ends before it starts');
    end
    for f = {'exclusionSec', 'pairZoneSec', 'pairMinSec'}
        if ~isfield(rule, f{1}) || ~isscalar(rule.(f{1})) || ~(rule.(f{1}) >= 0)
            error('night6:spanRule', 'rule.%s is required (seconds, >= 0)', f{1});
        end
    end
    dist = inf(n, 1);
    near = zeros(n, 1);
    inside = false(n, 1);
    for k = 1:size(spans, 1)
        a = spans(k, 1); b = spans(k, 2);
        d = zeros(n, 1);
        d(peaks < a) = (a - peaks(peaks < a)) / fs;
        d(peaks > b) = (peaks(peaks > b) - b) / fs;
        inside = inside | (peaks >= a & peaks <= b);
        closer = d < dist;
        dist(closer) = d(closer);
        near(closer) = k;
    end
    b = dist < rule.exclusionSec;
    zone = dist <= rule.pairZoneSec;
    [inPair, pairs] = pair_rule(peaks, true(n, 1), zone, fs, rule.pairMinSec);
    [~, pairsAfterB] = pair_rule(peaks, ~b, zone, fs, rule.pairMinSec);
    K = struct();
    K.kept = ~b & ~inPair;
    K.dropped_b = b;
    K.dropped_c = inPair & ~b;
    K.in_c_pair = inPair;
    K.inside = inside;
    K.dist_s = dist;
    K.nearest_span = near;
    K.in_zone = zone;
    K.pairs_c = pairs;
    K.pairs_c_spans = reshape(near(pairs), [], 2);
    K.pairs_c_after_b = pairsAfterB;
    K.n = n;
    K.n_kept = nnz(K.kept);
    K.n_b = nnz(K.dropped_b);
    K.n_c = nnz(K.dropped_c);
end

function [inPair, pairs] = pair_rule(peaks, use, zone, fs, minSec)
% (c) on the peaks marked use, consecutive among them; indices refer to peaks.
    idx = find(use);
    inPair = false(numel(peaks), 1);
    pairs = zeros(0, 2);
    for j = 1:numel(idx) - 1
        i1 = idx(j); i2 = idx(j + 1);
        if (peaks(i2) - peaks(i1)) / fs < minSec && zone(i1) && zone(i2)
            inPair([i1 i2]) = true;
            pairs(end + 1, :) = [i1 i2]; %#ok<AGROW>
        end
    end
end
