function check_slow_wave_span_rule(caseFile, outFile)
% CHECK_SLOW_WAVE_SPAN_RULE  Harness for tests/test_night6_slow_wave_span_rule.py.
%
% RULING 2026-10-12 (b)-(d), in ONE MATLAB process:
%   rule     night6_slow_wave_span_rule on synthetic peak lists (epoch rows) and spans, with
%            the declared rule (night6_slow_wave_settings().spanRule): kept / (b) / (c) per
%            case, the (c) pairs and the after-(b) reading
%   rate     night6_slow_wave_rate_series on a hand-made mask and peak list: the rate at
%            chosen centres (the expectation is computed independently in Python)
%   zone     night6_slow_wave_zone_mask at the 10 s boundary
%   apply    night6_slow_wave_apply_rule on a hand-made file of hers (full rate and
%            decimated by 78): kept peaks, rate over the zone-excluded clean time
%   refused  unsorted peaks, a reversed span, a missing rule field (error ids)
% Writes what it measured as JSON.
    C = jsondecode(fileread(caseFile));
    addpath(C.night6);
    W = night6_slow_wave_settings();
    out = struct('rule', W.spanRule, 'cases', struct());
    for f = fieldnames(C.cases)'
        c = C.cases.(f{1});
        sp = reshape(double(c.spans), [], 2);
        K = night6_slow_wave_span_rule(double(c.peaks), sp, C.fs, W.spanRule);
        out.cases.(f{1}) = struct('kept', find(K.kept)', 'b', find(K.dropped_b)', ...
            'c', find(K.dropped_c)', 'inside', find(K.inside)', ...
            'pairs', K.pairs_c, 'pairs_after_b', K.pairs_c_after_b, ...
            'pairs_spans', K.pairs_c_spans, 'n', K.n, 'n_kept', K.n_kept, ...
            'n_b', K.n_b, 'n_c', K.n_c, 'dist', K.dist_s');
    end
    R = C.rate;
    inv = false(R.n, 1); clean = true(R.n, 1);
    for k = 1:size(R.invalid, 1), inv(R.invalid(k, 1):R.invalid(k, 2)) = true; end
    for k = 1:size(R.unclean, 1), clean(R.unclean(k, 1):R.unclean(k, 2)) = false; end
    clean = clean & ~inv;
    [rate, rt] = night6_slow_wave_rate_series(R.locs, inv, clean, R.fs, 60, 30, 3);
    out.rate = struct('rate', rate', 'time', rt');
    % --- zone mask (rule (b) clean time) ------------------------------------------------
    Z = C.zone;
    out.zone = struct('mask', night6_slow_wave_zone_mask(double(Z.rows), ...
        reshape(double(Z.spans), [], 2), Z.fs, W.spanRule.exclusionSec)');
    % --- pre-span diagnostic on its own -------------------------------------------------
    out.prespan = struct();
    for f = fieldnames(C.prespan)'
        c = C.prespan.(f{1});
        P = night6_slow_wave_pre_span_peaks(double(c.rows), double(c.amps), ...
            reshape(double(c.spans), [], 2), c.fs, W.preSpanDiagnostic);
        out.prespan.(f{1}) = struct('n', P.n, 'row', P.row', 'dist_s', P.dist_s', ...
            'span', P.span', 'defined', P.defined, 'median', P.median);
    end
    out.p2t = struct();
    for f = fieldnames(C.p2t)'
        c = C.p2t.(f{1});
        [a, tr] = night6_slow_wave_peak_to_trough(double(c.ys), double(c.detected), ...
                                                  double(c.kept));
        out.p2t.(f{1}) = struct('amp', {num2cell(a')}, 'trough', {num2cell(tr')});
    end
    out.prespan_refused = attempt(@() night6_slow_wave_pre_span_peaks([1; 2], 1, [], 10, ...
        W.preSpanDiagnostic));
    % --- apply rule on a hand-made file of hers, full and decimated --------------------
    out.apply = struct();
    for f = fieldnames(C.apply)'
        out.apply.(f{1}) = apply_case(C.apply.(f{1}), fullfile(C.work, f{1}), W);
    end
    % --- the slow-wave caveat checked against a mask provenance --------------------------
    out.extent = struct();
    for f = fieldnames(C.extent)'
        r = struct('error_id', '', 'candidate_checked', '', 'floor_checked', '', 'statement', '');
        try
            E = night6_slow_wave_extent(jsondecode(C.extent.(f{1})), W, ...
                                        logical(C.extent_pilot.(f{1})));
            r.candidate_checked = E.candidate_checked;
            r.floor_checked = E.floor_checked;
            r.statement = E.statement;
        catch ME
            r.error_id = ME.identifier;
        end
        out.extent.(f{1}) = r;
    end
    % --- the span rule's part of the resume key (review of 3ce6575) ---------------------
    out.resume = struct();
    for f = fieldnames(C.resume)'
        c = C.resume.(f{1});
        [tf, why] = night6_span_rule_resumable(jsondecode(c.record), c.subset, ...
                                               W.spanRule.name);
        out.resume.(f{1}) = struct('tf', tf, 'why', why);
    end
    out.refused = struct( ...
        'unsorted', attempt(@() night6_slow_wave_span_rule([5; 3], zeros(0, 2), 1, W.spanRule)), ...
        'reversed', attempt(@() night6_slow_wave_span_rule(3, [9 2], 1, W.spanRule)), ...
        'no_rule', attempt(@() night6_slow_wave_span_rule(3, [1 2], 1, struct())));
    fid = fopen(outFile, 'w', 'n', 'UTF-8');
    fwrite(fid, jsonencode(out), 'char');
    fclose(fid);
end

function r = apply_case(c, d, W)
% A file as night6_keep_slow_wave writes it: her rate from her own peaks (so the replica
% check holds), then night6_slow_wave_apply_rule; what it left in the file.
    r = struct('error', '');
    try
        mkdir(d);
        n = c.n; fs = c.fs;
        inv = false(n, 1); edge = false(n, 1);
        iv = reshape(double(c.invalid), [], 2); ed = reshape(double(c.edge), [], 2);
        for k = 1:size(iv, 1), inv(iv(k, 1):iv(k, 2)) = true; end
        for k = 1:size(ed, 1), edge(ed(k, 1):ed(k, 2)) = true; end
        called = double(c.called(:));
        S = struct('fs', fs, 'invalidMask', inv, 'edgeMask', edge, 'rateWinSec', 60, ...
                   'minStretchSec', 30, 'minPeaksInStretch', 3, 'channel', 'ANT1');
        ys = zeros(n, 1);
        ys(called) = double(c.amps(:));   % her trace at her peaks (volts)
        tr = reshape(double(c.troughs), [], 2);   % [row value]: troughs placed by the case
        ys(tr(:, 1)) = tr(:, 2);
        ys(inv) = NaN;
        S.slowWaveTimeSeries = ys;
        [S.slowWaveRateSeries, S.slowWaveRateTime] = night6_slow_wave_rate_series(called, ...
            inv, ~inv & ~edge, fs, 60, 30, 3);
        S.avgSlowWave = mean(S.slowWaveRateSeries, 'omitnan');
        S.sw_implausibleFraction = mean(60 ./ (diff(called) / fs) > 8);
        if c.factor == 1
            S.slowWavePeakLocs = called;
        else
            S.slowWavePeakLocs = (called - 1) * c.factor + 1;
            S.decimation = struct('factor', c.factor, 'peak_locs_called', called);
        end
        save(fullfile(d, 'e0_slowWaves_ANT1.mat'), '-struct', 'S');
        d0 = [d '_nodiag'];   % the same file, the diagnostic off
        mkdir(d0);
        save(fullfile(d0, 'e0_slowWaves_ANT1.mat'), '-struct', 'S');
        A = night6_slow_wave_apply_rule(d, {'e0_slowWaves_ANT1.mat'}, ...
            reshape(double(c.spans), [], 2), c.fs_epoch, W);
        W0 = W; W0.preSpanDiagnostic = [];
        night6_slow_wave_apply_rule(d0, {'e0_slowWaves_ANT1.mat'}, ...
            reshape(double(c.spans), [], 2), c.fs_epoch, W0);
        F = load(fullfile(d, 'e0_slowWaves_ANT1.mat'));
        F0 = load(fullfile(d0, 'e0_slowWaves_ANT1.mat'));
        Fx = F; Fx.slow_wave_span_rule = rmfield(Fx.slow_wave_span_rule, 'pre_span_peaks');
        r.identical_without_diagnostic = isequaln(Fx, F0);
        r.no_diag_field = ~isfield(F0.slow_wave_span_rule, 'pre_span_peaks');
        r.rate_bits_equal = isequal(typecast(F.slowWaveRateSeries(:), 'uint64'), ...
                                    typecast(F0.slowWaveRateSeries(:), 'uint64'));
        P = F.slow_wave_span_rule.pre_span_peaks;
        r.pre = struct('n', P.n, 't_epoch_s', P.t_epoch_s', 'amplitude_uV', P.amplitude_uV', ...
                       'dist_s', P.dist_s', 'median', P.median, ...
                       'trough_t_epoch_s', P.trough_t_epoch_s', 'kind', P.amplitude_kind);
        r.caveat = F.slow_wave_span_rule.caveat;
        r.record_caveat = A.ANT1.caveat;
        r.record_pre_n = A.ANT1.pre_span_peaks.n;
        r.kept_epoch = F.slowWavePeakLocs(:)';
        if c.factor == 1
            r.kept_called = F.slowWavePeakLocs(:)';
        else
            r.kept_called = F.decimation.peak_locs_called(:)';
        end
        r.rate = F.slowWaveRateSeries(:)';
        r.avg = F.avgSlowWave;
        r.n_detected = A.ANT1.n_detected;
        r.n_dropped_b = A.ANT1.n_dropped_b;
        r.n_dropped_c = A.ANT1.n_dropped_c;
    catch ME
        r.error = sprintf('%s: %s', ME.identifier, ME.message);
    end
end

function id = attempt(f)
    id = '';
    try
        f();
    catch ME
        id = ME.identifier;
    end
end
