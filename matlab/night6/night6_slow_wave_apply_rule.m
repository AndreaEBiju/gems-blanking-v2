function A = night6_slow_wave_apply_rule(outDir, files, spans, fsEpoch, rule)
% NIGHT6_SLOW_WAVE_APPLY_RULE  RULING 2026-10-12 (b)-(d) on each kept slow-wave file of one run.
%
%   A = night6_slow_wave_apply_rule(outDir, files, spans, fsEpoch, rule)
%
%   files    the <base>_slowWaves_<channel>.mat files night6_keep_slow_wave wrote
%   spans    the run's masked spans, [nSeg x 2] 1-based inclusive EPOCH rows (the NaN of the
%            input; passed to her function as NaN only, RULING 2026-10-12 (a))
%   fsEpoch  the epoch's rate; rule = night6_slow_wave_settings().spanRule
%
% Per file, in place (the file was just written by night6_keep_slow_wave, before any trim):
%   1. night6_slow_wave_span_rule on her peaks (epoch rows) and the spans: kept / (b) / (c);
%   2. her rate loop replayed on HER peaks and HER clean mask must give her
%      slowWaveRateSeries bit for bit, and her implausible fraction likewise
%      ('night6:rateReplica') - the replica is checked on every call before it is used;
%   3. (d) slowWavePeakLocs (and decimation.peak_locs_called) become the kept peaks,
%      slowWaveRateSeries the rate from the kept peaks (her clean mask unchanged),
%      avgSlowWave its mean (omitnan, her expression), sw_implausibleFraction from the kept
%      peaks (her expression);
%   4. everything she detected is kept beside it in the added variable
%      slow_wave_span_rule: the rule, her peaks, rate, average and implausible fraction,
%      the rows dropped by (b) (inside a span listed too) and by (c), the (c) pairs, the
%      pairs the other reading would give, and - a SENSITIVITY, never an output - the kept
%      peaks' rate with the < exclusionSec zones also left out of the clean time.
% A.<channel> = counts and averages for the run record.
    A = struct();
    for f = files(:)'
        p = fullfile(outDir, f{1});
        S = load(p);
        fsc = double(S.fs);
        peaks = double(S.slowWavePeakLocs(:));
        dec = isfield(S, 'decimation');
        if dec
            called = double(S.decimation.peak_locs_called(:));
            if ~isequal((called - 1) * S.decimation.factor + 1, peaks)
                error('night6:spanRule', '%s: peak rows do not map to her called rows', f{1});
            end
        else
            called = peaks;
        end
        K = night6_slow_wave_span_rule(peaks, spans, fsEpoch, rule);
        inv = logical(S.invalidMask(:));
        cleanHer = ~inv & ~logical(S.edgeMask(:));
        args = {fsc, S.rateWinSec, S.minStretchSec, S.minPeaksInStretch};
        rateHer = night6_slow_wave_rate_series(called, inv, cleanHer, args{:});
        if ~isequaln(rateHer, double(S.slowWaveRateSeries(:))) ...
                || ~isequaln(implausible(called, fsc), double(S.sw_implausibleFraction))
            error('night6:rateReplica', ['%s: the rate replica does not reproduce her ' ...
                  'slowWaveRateSeries / sw_implausibleFraction from her own peaks'], f{1});
        end
        keptCalled = called(K.kept);
        rateKept = night6_slow_wave_rate_series(keptCalled, inv, cleanHer, args{:});
        zone5 = near_rows(numel(inv), dec, S, spans, fsEpoch, rule.exclusionSec);
        rateZone = night6_slow_wave_rate_series(keptCalled, inv, cleanHer & ~zone5, args{:});

        R = struct('rule', rule, 'spans_epoch_rows', spans, 'fs_epoch', fsEpoch, ...
                   'peaks_detected', peaks, 'peaks_detected_called', called, ...
                   'rate_detected', S.slowWaveRateSeries, 'avg_detected', S.avgSlowWave, ...
                   'implausible_detected', S.sw_implausibleFraction, ...
                   'dropped_b', peaks(K.dropped_b), 'dropped_b_inside', peaks(K.inside), ...
                   'dropped_c', peaks(K.dropped_c), 'pairs_c', reshape(peaks(K.pairs_c), [], 2), ...
                   'pairs_c_nearest_span', K.pairs_c_spans, ...
                   'pairs_c_if_after_b', reshape(peaks(K.pairs_c_after_b), [], 2), ...
                   'dist_s', K.dist_s, 'n_detected', K.n, 'n_kept', K.n_kept, ...
                   'n_dropped_b', K.n_b, 'n_dropped_c', K.n_c, ...
                   'rate_kept_zone_excluded', rateZone, ...
                   'avg_kept_zone_excluded', mean(rateZone, 'omitnan'), ...
                   'zone_excluded_note', ['SENSITIVITY only: the kept peaks'' rate with the ' ...
                                          '< exclusionSec zones also removed from the clean ' ...
                                          'time (the ruling counts peaks only)'], ...
                   'replica_check', 'her rate and implausible fraction reproduced bit for bit');
        S.slowWavePeakLocs = peaks(K.kept);
        if dec, S.decimation.peak_locs_called = keptCalled; end
        S.slowWaveRateSeries = cast(rateKept, 'like', S.slowWaveRateSeries);
        S.avgSlowWave = mean(S.slowWaveRateSeries, 'omitnan');
        S.sw_implausibleFraction = implausible(keptCalled, fsc);
        S.slow_wave_span_rule = R;
        save(p, '-struct', 'S');
        ch = matlab.lang.makeValidName(char(S.channel));
        A.(ch) = struct('file', f{1}, 'n_detected', K.n, 'n_kept', K.n_kept, ...
                        'n_dropped_b', K.n_b, 'n_dropped_b_inside', nnz(K.inside), ...
                        'n_dropped_c', K.n_c, 'n_pairs_c', size(K.pairs_c, 1), ...
                        'n_pairs_c_if_after_b', size(K.pairs_c_after_b, 1), ...
                        'avg_detected', R.avg_detected, 'avg_kept', S.avgSlowWave, ...
                        'avg_kept_zone_excluded', R.avg_kept_zone_excluded);
    end
end

function v = implausible(locs, fs)
% her sw_implausibleFraction (slowWaveAnalysis_new.m:207-213): NaN under two peaks
    v = NaN;
    if numel(locs) >= 2
        v = mean(60 ./ (diff(locs) / fs) > 8);
    end
end

function z = near_rows(Nc, dec, S, spans, fsEpoch, sec)
% rows (at the rate she ran at) whose epoch row is < sec from a span or inside one
% (sensitivity only)
    if dec
        rows = ((1:Nc)' - 1) * S.decimation.factor + 1;
    else
        rows = (1:Nc)';
    end
    z = false(Nc, 1);
    for k = 1:size(spans, 1)   % distance < sec, as rule (b): strict on both sides
        z = z | (rows > spans(k, 1) - sec * fsEpoch & rows < spans(k, 2) + sec * fsEpoch);
    end
end
