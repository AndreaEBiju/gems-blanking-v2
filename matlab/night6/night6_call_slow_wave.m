function S = night6_call_slow_wave(X, fs, W, outDir, label, base, signals, keep, maskSignal, ...
                                   spans, rate, figs)
% NIGHT6_CALL_SLOW_WAVE  One slowWaveAnalysis_new run, masked spans as NaN only + the span rule.
%
%   S = night6_call_slow_wave(X, fs, W, outDir, label, base, signals, keep, maskSignal, ...
%                             spans, rate, figs)
%
%   X           N x 3 ANT input, volts, maskSignal's slow_wave mask NaN on every column
%   fs          the epoch's rate
%   W           the slow-wave parameters (night6_slow_wave_settings, = params().slow_wave)
%   spans       maskSignal's slow_wave spans, 1-based inclusive epoch rows - exactly the
%               NaN of X (asserted, 'night6:blankIdx')
%   rate        the declared slow-wave rate (night6_slow_wave_rates; REQUIRED)
%   keep        the channels whose own mask equals maskSignal's (the (j) 5 (a) shortcut:
%               a shared call passes the SHARED spans, which are each kept channel's own)
%
% RULING 2026-10-12 (Andrea; replaces RULING 2026-10-09 (c) 6 for slow wave; PILOT):
% (a) the masked spans go to her function as NaN ONLY - blankIdx = [] - so her
%     edgeBufferSec (W.edgeBufferSec, 15 s) guards the file's own start and end and no
%     masked span; (b)-(d) are applied to her detected peaks after the call, per kept
%     channel, by night6_slow_wave_apply_rule (W.spanRule): peaks inside a span or
%     < exclusionSec (10 s) from one are ignored and that time is left out of the rate's
%     clean time, both peaks of a consecutive detected pair < 7 s apart within 15 s of a
%     span are dropped, and the rate, its average and the implausible fraction are
%     recomputed from the kept peaks. NOT covered by the harm test (W.spanRule.harm_test).
%
% 'decimated78': X is decimated first by night6_decimate_masked (the decimation check's
% own logic; a decimated row is NaN if any source row is), her function runs at fs / 78,
% and night6_keep_slow_wave maps her peak rows back to epoch rows at fs (decimated row j =
% epoch row (j - 1) 78 + 1). Her seconds (t, slowWaveRateTime) are already epoch time. The
% rule judges distances in EPOCH rows against the source spans.
%
% S.slow_wave  night6_keep_slow_wave's record;  S.span_rule  per kept channel, the counts
% (detected, kept, dropped by (b), by (c)) and the averages before and after;
% S.blank_idx  the form, count, rate and rule;  S.slow_wave_rate  the declared rate, its
% factors and the rate she ran at;  S.caveats  night6_slow_wave_caveats(W, R).
    N = size(X, 1);
    [~, R] = night6_slow_wave_rates(rate, fs);
    cover = false(N, 1);
    for k = 1:size(spans, 1), cover(spans(k, 1):spans(k, 2)) = true; end
    if ~isequal(isnan(X), repmat(cover, 1, size(X, 2)))
        error('night6:blankIdx', ['%s: the masked spans must be exactly the input''s NaN: ' ...
              'the input''s NaN differs from %s''s %d span(s) at %d row(s)'], label, ...
              maskSignal, size(spans, 1), nnz(any(isnan(X), 2) ~= cover));
    end
    dec = [];
    if isempty(R.factors)
        Xc = X; fsc = fs;
    else
        [Xc, fsc, mapped] = night6_decimate_masked(X, fs, R.factors, spans);
        dec = struct('factor', R.factor, 'factors', R.factors, 'fs_source', fs, ...
                     'fs_called', fsc, 'n_source', N, 'n_called', size(Xc, 1), ...
                     'spans_source', spans, 'spans_called', mapped, ...
                     'rule', ['night6_decimate_masked (= the decimation check''s decimate_masked ' ...
                              'and map_spans): zero-phase fir1(20 r, 0.8 / r) filtfilt per stage ' ...
                              'over a temporary linear fill, reverted; a decimated row is NaN ' ...
                              'if any source row is; decimated row j = epoch row (j - 1) R + 1']);
    end
    blank = zeros(0, 2);   % RULING 2026-10-12 (a): NaN only, never blankIdx
    d0 = dir(outDir);
    slowWaveAnalysis_new(Xc, W.lowPassOn, W.lowPassCutoff, W.lowPassOrder, fsc, ...
        W.smoothWindow, figs, outDir, label, blank, W.edgeBufferSec);
    d1 = dir(outDir);
    S.slow_wave = night6_keep_slow_wave(outDir, label, base, signals, keep, maskSignal, ...
                                        setdiff({d1.name}, {d0.name}), dec);
    S.span_rule = night6_slow_wave_apply_rule(outDir, S.slow_wave.files, spans, fs, W.spanRule);
    S.blank_idx = struct('form', 'none: blankIdx = [] (RULING 2026-10-12 (a))', ...
        'n_spans', 0, 'n_masked_spans', size(spans, 1), 'fs', fsc, 'mask_signal', maskSignal, ...
        'masked_spans', 'the run mask''s NaN spans, passed as NaN only', ...
        'edge_buffer_s', W.edgeBufferSec, 'edge_buffer_scope', W.edgeBufferScope, ...
        'span_rule', W.spanRule.name);
    S.slow_wave_rate = struct('name', R.name, 'factors', R.factors, 'factor', R.factor, ...
                              'fs_called', fsc);
    S.caveats = night6_slow_wave_caveats(W, R);   % built from the settings used (fix 5)
end
