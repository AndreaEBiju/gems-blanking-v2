function check_slow_wave_rate(caseFile, outFile)
% CHECK_SLOW_WAVE_RATE  Harness for tests/test_night6_slow_wave.py (gems-blanking-v2).
%
% RULING 2026-10-09 (c) 6, in ONE MATLAB process:
%   refusals   night6_slow_wave_rates / night6_check_decimation on missing, unknown and
%              non-divisor declarations (the error id of each attempt)
%   decimate   night6_decimate_masked against the decimation check's own decimate_masked
%              and map_spans (reference_*, below: copied from decim_blank.m), bit for bit,
%              on an input with spans at the block boundaries; the NaN rows and spans
%   keep       night6_keep_slow_wave on a hand-made file of hers with peak rows 1, 2, 40,
%              decimated by 78: the epoch rows they map to
%   calls      night6_call_slow_wave with her real slowWaveAnalysis_new, full rate and
%              decimated: a shared call keeping two channels and a per-channel call, what
%              her saved files hold (blankIdx, invalidMask, edgeMask, peaks, decimation),
%              and the refusal of spans that are not the input's NaN
% Writes what it measured as JSON.
    C = jsondecode(fileread(caseFile));
    addpath(C.processing_new); addpath(C.night6);
    out = struct();
    out.refusals = struct( ...
        'none', attempt(@() night6_slow_wave_rates('')), ...
        'unknown', attempt(@() night6_slow_wave_rates('decimated100')), ...
        'f7', attempt(@() night6_check_decimation(7, [])), ...
        'f4', attempt(@() night6_check_decimation([2 2], [])), ...
        'f78_at_1000', attempt(@() night6_check_decimation([13 6], 1000)), ...
        'f1', attempt(@() night6_check_decimation(1, [])), ...
        'f78', attempt(@() night6_check_decimation([13 6], 24414.0625)), ...
        'f313', attempt(@() night6_check_decimation(313, [])), ...
        'decimate_f7', attempt(@() night6_decimate_masked(zeros(100, 3), 24414.0625, 7)));
    [table, Rf] = night6_slow_wave_rates('decimated78', 24414.0625);
    out.table = struct('names', {{table.name}}, 'factor', Rf.factor, 'factors', Rf.factors);
    [~, R0] = night6_slow_wave_rates('full');
    out.full_factor = R0.factor;

    % --- decimate: against the decimation check's own functions ----------------------
    D = load(C.decimate.input_file);
    spans = double(D.spans);
    X = D.X;
    for j = 1:size(spans, 1), X(spans(j, 1):spans(j, 2), :) = NaN; end
    [Y, fsd, spd] = night6_decimate_masked(X, D.fs, [13 6], spans);
    [Yr, fsr] = reference_decimate_masked(X, D.fs, [13 6]);
    spr = reference_map_spans(spans, 78, size(Yr, 1));
    out.decimate = struct('equal_to_check', isequaln(Y, Yr) && fsd == fsr, ...
        'spans_equal_to_check', isequal(spd, spr), 'spd', spd, 'fsd', fsd, ...
        'nd', size(Y, 1), 'nan_rows', find(isnan(Y(:, 1)))', ...
        'nan_all_columns_equal', isequal(isnan(Y), repmat(isnan(Y(:, 1)), 1, 3)), ...
        'n_nan_input', nnz(isnan(X(:, 1))));
    % a span that is not the NaN is refused
    Xb = X; Xb(400, 2) = NaN;   % block 6: no span there
    out.decimate.refused = attempt(@() night6_decimate_masked(Xb, D.fs, [13 6], spans));

    % --- keep: peak rows mapped back to epoch rows (1-sample boundary) ----------------
    kd = fullfile(C.work, 'keep');
    mkdir(kd);
    slowWaveTimeSeries = zeros(50, 3); slowWaveRateSeries = zeros(5, 3); %#ok<NASGU>
    avgSlowWave = [1 2 3]; slowWavePeakLocs = {[1; 2; 40], [3; 4], 5}; %#ok<NASGU>
    sw_implausibleFraction = [0 0 0]; fs = D.fs / 78; t = (0:49)' / fs; %#ok<NASGU>
    save(fullfile(kd, 'k_slowWaves.mat'), 'slowWaveTimeSeries', 'slowWaveRateSeries', ...
         'avgSlowWave', 'slowWavePeakLocs', 'sw_implausibleFraction', 'fs', 't');
    dec = struct('factor', 78, 'factors', [13 6], 'fs_source', D.fs, 'fs_called', D.fs / 78);
    night6_keep_slow_wave(kd, 'k', 'e0', {'ANT1', 'ANT2', 'ANT3'}, {'ANT1'}, 'ANT1', ...
                          {'k_slowWaves.mat'}, dec);
    K = load(fullfile(kd, 'e0_slowWaves_ANT1.mat'));
    out.keep = struct('locs', K.slowWavePeakLocs(:)', ...
                      'called', K.decimation.peak_locs_called(:)', 'fs', K.fs);

    % --- calls: her real function, through night6_call_slow_wave ---------------------
    S = load(C.calls.input_file);
    W = struct('lowPassOn', true, 'lowPassCutoff', 0.15, 'lowPassOrder', 2, ...
               'smoothWindow', 5, 'edgeBufferSec', 15);
    signals = {'ANT1', 'ANT2', 'ANT3'};
    out.calls = struct();
    for rate = {'full', 'decimated78'}
        for run = {'shared', 'own'}
            if strcmp(run{1}, 'shared')
                sp = double(S.spans_shared); keep = {'ANT1', 'ANT3'}; ms = 'ANT1';
            else
                sp = double(S.spans_ant2); keep = {'ANT2'}; ms = 'ANT2';
            end
            Xc = S.X;
            for j = 1:size(sp, 1), Xc(sp(j, 1):sp(j, 2), :) = NaN; end
            d = fullfile(C.work, [rate{1} '_' run{1}]);
            mkdir(d);
            rec = struct('error', '');
            try
                r = night6_call_slow_wave(Xc, S.fs, W, d, sprintf('e0_swm_%s', ms), 'e0', ...
                                          signals, keep, ms, sp, rate{1}, false);
                rec.blank_idx = r.blank_idx;
                rec.rate = r.slow_wave_rate;
                rec.files = r.slow_wave.files;
                rec.kept = struct();
                for k = keep(:)'
                    F = load(fullfile(d, sprintf('e0_slowWaves_%s.mat', k{1})));
                    e = struct('blankIdx', F.blankIdx, 'fs', F.fs, ...
                               'invalid_runs', runs(F.invalidMask), ...
                               'edge_runs', runs(F.edgeMask), 'n', numel(F.invalidMask), ...
                               'peaks', F.slowWavePeakLocs(:)', 'channel', F.channel);
                    if isfield(F, 'decimation')
                        e.peaks_called = F.decimation.peak_locs_called(:)';
                        e.decimation_factor = F.decimation.factor;
                        e.blankIdx_source = F.decimation.blankIdx_source;
                    end
                    rec.kept.(k{1}) = e;
                end
            catch ME
                rec.error = sprintf('%s: %s', ME.identifier, ME.message);
            end
            out.calls.([strrep(rate{1}, '78', '') '_' run{1}]) = rec;
        end
    end
    % --- trim (B) of a DECIMATED run's file: her rows at her rate, the cut in epoch rows --
    T = C.trim;
    RS = night6_recovery_start(T.starts_file);
    [~, rec] = night6_recovery_lead_in(RS, T.session, 'stim_recovery', 0, size(S.X, 1), ...
                                       S.fs, {'slow_wave'}, 'mask_to_electrical_drop_outputs');
    d = fullfile(C.work, 'decimated78_own');
    f = 'e0_slowWaves_ANT2.mat';
    B = load(fullfile(d, f));
    tr = struct('error', '');
    try
        info = night6_recovery_trim_outputs(d, {f}, {'slow_wave'}, rec, RS.outputTimes, S.fs);
        A = load(fullfile(d, f));
        tr.n_files = numel(info.files);
        tr.rate_t = B.slowWaveRateTime(:)';
        tr.fs = B.fs;
        tr.invalid_runs = runs(B.invalidMask);
        tr.n = numel(B.invalidMask);
        tr.rate_win = B.rateWinSec;
        tr.fraction = A.slowWaveRateSeries_validFraction(:)';
        tr.peaks_before = B.slowWavePeakLocs(:)';
        tr.peaks_after = A.slowWavePeakLocs(:)';
        tr.cuts = cellfun(@(c) struct('cut', c.cut, 'rows', c.output_rows_before_start), ...
                          rec.cuts, 'UniformOutput', false);
    catch ME
        tr.error = sprintf('%s: %s', ME.identifier, ME.message);
    end
    out.trim = tr;
    Xbad = S.X;   % the spans say masked, the input is not: refused
    d = fullfile(C.work, 'bad'); mkdir(d);
    out.calls.not_the_nan = attempt(@() night6_call_slow_wave(Xbad, S.fs, W, d, 'e0_swm_ANT1', ...
        'e0', signals, {'ANT1'}, 'ANT1', double(S.spans_shared), 'full', false));
    fid = fopen(outFile, 'w', 'n', 'UTF-8');
    fwrite(fid, jsonencode(out), 'char');
    fclose(fid);
end

function r = runs(b)
    e = diff([false; b(:); false]);
    r = [find(e == 1), find(e == -1) - 1];
end

function id = attempt(f)
    id = '';
    try
        f();
    catch ME
        id = ME.identifier;
    end
end

% --- the decimation check's own functions, copied from decim_blank.m (2026-10-09) -----
function [Y, fsd] = reference_decimate_masked(X, fs, factors)
% Unchanged from S\night6_1009\sw_decim_check.m.
    Y = X; fsd = fs;
    for r = factors
        bad = isnan(Y);
        F = Y;
        for c = 1:size(F, 2)
            F(:, c) = fillmissing(F(:, c), 'linear', 'EndValues', 'nearest');   % temporary
        end
        b = fir1(20 * r, 0.8 / r);
        F = filtfilt(b, 1, F);
        n = floor(size(F, 1) / r);
        Y = F(1:r:n * r, :);
        blk = reshape(bad(1:n * r, :), r, n, size(bad, 2));
        Y(squeeze(any(blk, 1))) = NaN;                                           % reverted
        fsd = fsd / r;
    end
end

function sd = reference_map_spans(sp, r, Nd)
% Any-source rule: decimated sample j (1-based) covers source samples (j-1)r+1 .. jr.
    if isempty(sp), sd = zeros(0, 2); return, end
    sd = [floor((sp(:, 1) - 1) / r) + 1, floor((sp(:, 2) - 1) / r) + 1];
    sd = sd(sd(:, 1) <= Nd, :);
    sd(:, 2) = min(sd(:, 2), Nd);
end
