function A = night6_slow_wave_apply_rule(outDir, files, spans, fsEpoch, W)
% NIGHT6_SLOW_WAVE_APPLY_RULE  RULING 2026-10-12 (b)-(d) on each kept slow-wave file of one run.
%
%   A = night6_slow_wave_apply_rule(outDir, files, spans, fsEpoch, W)
%
%   files    the <base>_slowWaves_<channel>.mat files night6_keep_slow_wave wrote
%   spans    the run's masked spans, [nSeg x 2] 1-based inclusive EPOCH rows (the NaN of the
%            input; passed to her function as NaN only, RULING 2026-10-12 (a))
%   fsEpoch  the epoch's rate
%   W        night6_slow_wave_settings(): its spanRule, extent (the caveat every result
%            carries) and preSpanDiagnostic (W.preSpanDiagnostic = [] skips the diagnostic)
%
% Per file, in place (the file was just written by night6_keep_slow_wave, before any trim):
%   1. night6_slow_wave_span_rule on her peaks (epoch rows) and the spans: kept / (b) / (c);
%   2. her rate loop replayed on HER peaks and HER clean mask must give her
%      slowWaveRateSeries bit for bit, and her implausible fraction likewise
%      ('night6:rateReplica') - the replica is checked on every call before it is used;
%   3. (b) the rows less than rule.exclusionSec from a span (night6_slow_wave_zone_mask)
%      are left out of the clean time: clean = ~invalidMask & ~edgeMask & ~zone (her
%      invalidMask and edgeMask variables are NOT changed);
%   4. (d) slowWavePeakLocs (and decimation.peak_locs_called) become the kept peaks,
%      slowWaveRateSeries the rate from the kept peaks over that clean time, avgSlowWave its
%      mean (omitnan, her expression), sw_implausibleFraction from the kept peaks (her
%      expression);
%   5. everything she detected is kept beside it in the added variable
%      slow_wave_span_rule: the rule, her peaks, rate, average and implausible fraction,
%      the rows dropped by (b) (inside a span listed too) and by (c), the (c) pairs, the
%      pairs the after-(b) reading would give (reported), the zone mask's runs, and - for
%      comparison only - the kept peaks' rate with the zone time COUNTED as clean (v1),
%      W.extent.statement (slow wave is C1 at floor 7.42 and NOT covered by the harm test),
%      and the DIAGNOSTIC pre_span_peaks (night6_slow_wave_pre_span_peaks: kept peaks
%      within 25 s before a span start whose PEAK-TO-TROUGH amplitude
%      (night6_slow_wave_peak_to_trough) is > 2 x the channel's median). The diagnostic is computed last, from the kept
%      peaks only, and changes no other variable.
% A.<channel> = counts and averages for the run record.
    rule = W.spanRule;
    if ~isfield(rule, 'exclusionFromCleanTime') || ~islogical(rule.exclusionFromCleanTime)
        error('night6:spanRule', 'rule.exclusionFromCleanTime is required (logical)');
    end
    A = struct();
    for f = files(:)'
        p = fullfile(outDir, f{1});
        S = load(p);
        fsc = double(S.fs);
        peaks = double(S.slowWavePeakLocs(:));
        dec = isfield(S, 'decimation');
        nc = numel(S.invalidMask);
        if dec
            called = double(S.decimation.peak_locs_called(:));
            if ~isequal((called - 1) * S.decimation.factor + 1, peaks)
                error('night6:spanRule', '%s: peak rows do not map to her called rows', f{1});
            end
            rowsEpoch = ((1:nc)' - 1) * S.decimation.factor + 1;
        else
            called = peaks;
            rowsEpoch = (1:nc)';
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
        zone = night6_slow_wave_zone_mask(rowsEpoch, spans, fsEpoch, rule.exclusionSec);
        if rule.exclusionFromCleanTime
            cleanRule = cleanHer & ~zone;
        else
            cleanRule = cleanHer;
        end
        rateKept = night6_slow_wave_rate_series(keptCalled, inv, cleanRule, args{:});
        rateCounted = night6_slow_wave_rate_series(keptCalled, inv, cleanHer, args{:});

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
                   'zone_runs_called', runs(zone), ...
                   'zone_s', nnz(zone & ~inv) / fsc, ...
                   'clean_s_her', nnz(cleanHer) / fsc, 'clean_s_rule', nnz(cleanRule) / fsc, ...
                   'rate_kept_zone_counted', rateCounted, ...
                   'avg_kept_zone_counted', mean(rateCounted, 'omitnan'), ...
                   'zone_counted_note', ['COMPARISON only (pilot v1): the kept peaks'' rate with ' ...
                                         'the < exclusionSec zones counted as clean time'], ...
                   'replica_check', 'her rate and implausible fraction reproduced bit for bit', ...
                   'caveat', W.extent.statement, 'extent', W.extent);
        S.slowWavePeakLocs = peaks(K.kept);
        if dec, S.decimation.peak_locs_called = keptCalled; end
        S.slowWaveRateSeries = cast(rateKept, 'like', S.slowWaveRateSeries);
        S.avgSlowWave = mean(S.slowWaveRateSeries, 'omitnan');
        S.sw_implausibleFraction = implausible(keptCalled, fsc);
        if ~isempty(W.preSpanDiagnostic)   % DIAGNOSTIC: reads the kept peaks only
            [p2t, trCalled] = night6_slow_wave_peak_to_trough(S.slowWaveTimeSeries, ...
                called, keptCalled);   % peak-to-trough (Andrea 2026-10-12)
            if dec, trEpoch = (trCalled - 1) * S.decimation.factor + 1; else, trEpoch = trCalled; end
            R.pre_span_peaks = night6_slow_wave_pre_span_peaks(peaks(K.kept), ...
                p2t, spans, fsEpoch, W.preSpanDiagnostic, trEpoch);
        end
        S.slow_wave_span_rule = R;
        save(p, '-struct', 'S');
        ch = matlab.lang.makeValidName(char(S.channel));
        A.(ch) = struct('file', f{1}, 'n_detected', K.n, 'n_kept', K.n_kept, ...
                        'n_dropped_b', K.n_b, 'n_dropped_b_inside', nnz(K.inside), ...
                        'n_dropped_c', K.n_c, 'n_pairs_c', size(K.pairs_c, 1), ...
                        'n_pairs_c_if_after_b', size(K.pairs_c_after_b, 1), ...
                        'zone_s', R.zone_s, 'avg_detected', R.avg_detected, ...
                        'avg_kept', S.avgSlowWave, 'avg_kept_zone_counted', R.avg_kept_zone_counted, ...
                        'caveat', W.extent.statement);
        if isfield(R, 'pre_span_peaks')
            D = R.pre_span_peaks;
            A.(ch).pre_span_peaks = struct('n', D.n, 'median_amplitude_uV', D.median * 1e6, ...
                'defined', D.defined, 't_epoch_s', D.t_epoch_s, 'amplitude_uV', D.amplitude_uV, ...
                'dist_s', D.dist_s, 'trough_t_epoch_s', D.trough_t_epoch_s, ...
                'amplitude_kind', D.amplitude_kind, 'what', D.what);
        end
    end
end

function v = implausible(locs, fs)
% her sw_implausibleFraction (slowWaveAnalysis_new.m:207-213): NaN under two peaks
    v = NaN;
    if numel(locs) >= 2
        v = mean(60 ./ (diff(locs) / fs) > 8);
    end
end

function r = runs(b)
    e = diff([false; b(:); false]);
    r = [find(e == 1), find(e == -1) - 1];
end
