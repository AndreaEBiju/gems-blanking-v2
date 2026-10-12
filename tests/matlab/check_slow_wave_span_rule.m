function check_slow_wave_span_rule(caseFile, outFile)
% CHECK_SLOW_WAVE_SPAN_RULE  Harness for tests/test_night6_slow_wave_span_rule.py.
%
% RULING 2026-10-12 (b)-(d), in ONE MATLAB process:
%   rule     night6_slow_wave_span_rule on synthetic peak lists (epoch rows) and spans, with
%            the declared rule (night6_slow_wave_settings().spanRule): kept / (b) / (c) per
%            case, the (c) pairs and the after-(b) reading
%   rate     night6_slow_wave_rate_series on a hand-made mask and peak list: the rate at
%            chosen centres (the expectation is computed independently in Python)
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
    out.refused = struct( ...
        'unsorted', attempt(@() night6_slow_wave_span_rule([5; 3], zeros(0, 2), 1, W.spanRule)), ...
        'reversed', attempt(@() night6_slow_wave_span_rule(3, [9 2], 1, W.spanRule)), ...
        'no_rule', attempt(@() night6_slow_wave_span_rule(3, [1 2], 1, struct())));
    fid = fopen(outFile, 'w', 'n', 'UTF-8');
    fwrite(fid, jsonencode(out), 'char');
    fclose(fid);
end

function id = attempt(f)
    id = '';
    try
        f();
    catch ME
        id = ME.identifier;
    end
end
