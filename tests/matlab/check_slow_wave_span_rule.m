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
    % --- apply rule on a hand-made file of hers, full and decimated --------------------
    out.apply = struct();
    for f = fieldnames(C.apply)'
        out.apply.(f{1}) = apply_case(C.apply.(f{1}), fullfile(C.work, f{1}), W.spanRule);
    end
    out.refused = struct( ...
        'unsorted', attempt(@() night6_slow_wave_span_rule([5; 3], zeros(0, 2), 1, W.spanRule)), ...
        'reversed', attempt(@() night6_slow_wave_span_rule(3, [9 2], 1, W.spanRule)), ...
        'no_rule', attempt(@() night6_slow_wave_span_rule(3, [1 2], 1, struct())));
    fid = fopen(outFile, 'w', 'n', 'UTF-8');
    fwrite(fid, jsonencode(out), 'char');
    fclose(fid);
end

function r = apply_case(c, d, rule)
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
        A = night6_slow_wave_apply_rule(d, {'e0_slowWaves_ANT1.mat'}, ...
            reshape(double(c.spans), [], 2), c.fs_epoch, rule);
        F = load(fullfile(d, 'e0_slowWaves_ANT1.mat'));
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
