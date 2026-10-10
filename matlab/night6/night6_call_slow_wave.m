function S = night6_call_slow_wave(X, fs, W, outDir, label, base, signals, keep, maskSignal, ...
                                   spans, rate, figs)
% NIGHT6_CALL_SLOW_WAVE  One slowWaveAnalysis_new run, with its masked spans as blankIdx.
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
% RULING 2026-10-09 (c) 6, option (a): the masked spans go to her function as blankIdx AS
% WELL AS NaN, so her edgeBufferSec (15 s) guards every masked span - the electrical
% lead-in included - and not only the epoch ends. blankIdx is her documented form
% (slowWaveAnalysis_new.m:24): [nSeg x 2], 1-based inclusive sample rows at the fs passed;
% she rounds and clips it and marks it invalid (:99-107) and buffers it (:110-118). Her
% 15 s covers the ~8 s low-pass settling at a masked edge (8.17 s, task 13's impz of the
% 0.15 Hz order-2 low-pass; extent.recovery_start).
%
% 'decimated78': X is decimated first by night6_decimate_masked (the decimation check's
% own logic), blankIdx is the spans mapped by the same any-source rule, her function runs
% at fs / 78, and night6_keep_slow_wave maps her peak rows back to epoch rows at fs
% (decimated row j = epoch row (j - 1) 78 + 1). Her seconds (t, slowWaveRateTime) are
% already epoch time.
%
% S.slow_wave  night6_keep_slow_wave's record;  S.blank_idx  the form, count, rate and
% rule;  S.slow_wave_rate  the declared rate, its factors and the rate she ran at;
% S.caveats  night6_slow_wave_caveats(W, R): the (h) 1 setting, (h) 2 amplitude caveat and
% (f) 1 known properties, the setting text built from W and R (review 2026-10-10 fix 5).
    N = size(X, 1);
    [~, R] = night6_slow_wave_rates(rate, fs);
    cover = false(N, 1);
    for k = 1:size(spans, 1), cover(spans(k, 1):spans(k, 2)) = true; end
    if ~isequal(isnan(X), repmat(cover, 1, size(X, 2)))
        error('night6:blankIdx', ['%s: blankIdx must be exactly the masked spans: the input''s ' ...
              'NaN differs from %s''s %d span(s) at %d row(s)'], label, maskSignal, ...
              size(spans, 1), nnz(any(isnan(X), 2) ~= cover));
    end
    dec = [];
    if isempty(R.factors)
        Xc = X; fsc = fs; blank = spans;
    else
        [Xc, fsc, blank] = night6_decimate_masked(X, fs, R.factors, spans);
        dec = struct('factor', R.factor, 'factors', R.factors, 'fs_source', fs, ...
                     'fs_called', fsc, 'n_source', N, 'n_called', size(Xc, 1), ...
                     'blankIdx_source', spans, ...
                     'rule', ['night6_decimate_masked (= the decimation check''s decimate_masked ' ...
                              'and map_spans): zero-phase fir1(20 r, 0.8 / r) filtfilt per stage ' ...
                              'over a temporary linear fill, reverted; a decimated row is NaN ' ...
                              'if any source row is; decimated row j = epoch row (j - 1) R + 1']);
    end
    if isempty(blank), blank = zeros(0, 2); end
    d0 = dir(outDir);
    slowWaveAnalysis_new(Xc, W.lowPassOn, W.lowPassCutoff, W.lowPassOrder, fsc, ...
        W.smoothWindow, figs, outDir, label, blank, W.edgeBufferSec);
    d1 = dir(outDir);
    S.slow_wave = night6_keep_slow_wave(outDir, label, base, signals, keep, maskSignal, ...
                                        setdiff({d1.name}, {d0.name}), dec);
    S.blank_idx = struct('form', ['[nSeg x 2] double, 1-based inclusive sample rows at the ' ...
                                  'fs passed (slowWaveAnalysis_new.m:24; applied :99-118)'], ...
        'n_spans', size(blank, 1), 'fs', fsc, 'mask_signal', maskSignal, ...
        'equals', 'the run mask''s NaN spans (asserted), mapped by the any-source rule if decimated', ...
        'edge_buffer_s', W.edgeBufferSec, ...
        'covers', ['her 15 s edgeBufferSec covers the ~8 s low-pass settling (8.17 s, task 13) ' ...
                   'at every masked span (RULING 2026-10-09 (c) 6)']);
    S.slow_wave_rate = struct('name', R.name, 'factors', R.factors, 'factor', R.factor, ...
                              'fs_called', fsc);
    S.caveats = night6_slow_wave_caveats(W, R);   % built from the settings used (fix 5)
end
