function check_step1_masks(maskFile, signalFile, outFile)
% CHECK_STEP1_MASKS  Task 15 acceptance harness (gems-blanking-v2, tests/matlab).
%
%   check_step1_masks(maskFile, signalFile, outFile)
%
% Reads a mask file written by gems_blanking_v2.emit.handoff.write_mask_file and a
% synthetic signal file (variables y: N x nCh, fs, channelLabels), applies every
% blank_<consumer>_<signal> span set to its signal as NaN (1-based inclusive
% [start stop] rows, the blankSpans convention), runs Andrea's step1_bandpass on the
% spike consumer's channel with pipeline_params(), and writes what it measured as JSON.
%
% READ-ONLY with respect to processing_new: it only calls step1_bandpass and
% pipeline_params from the path. Nothing is written anywhere but outFile.

    % The step1_bandpass under test must be Andrea's, in processing_new - a stray copy
    % (e.g. beside this harness) would shadow it and the test would check nothing.
    where = which('step1_bandpass');
    [pdir, ~, ~] = fileparts(where);
    [~, leaf] = fileparts(pdir);
    assert(strcmp(leaf, 'processing_new'), 'step1_bandpass resolves to %s, not processing_new', where);

    M = load(maskFile);
    S = load(signalFile);
    fs = double(M.fs);
    N = double(M.nSamples);
    out = struct();
    out.fs_matches = abs(double(S.fs) - fs) < 1e-9;
    out.n_samples = N;

    names = fieldnames(M);
    masks = struct('name', {}, 'n_nan', {}, 'first', {}, 'last', {}, 'n_spans', {});
    for i = 1:numel(names)
        nm = names{i};
        if ~startsWith(nm, 'blank_') && ~startsWith(nm, 'notmeasured_')
            continue
        end
        sp = double(M.(nm));
        x = zeros(N, 1) + 1;          % a stand-in signal of ones: NaN placement only
        for k = 1:size(sp, 1)
            x(sp(k, 1):sp(k, 2)) = NaN;
        end
        bad = find(isnan(x));
        e = struct('name', nm, 'n_nan', numel(bad), 'first', [], 'last', [], ...
                   'n_spans', size(sp, 1));
        if ~isempty(bad)
            e.first = bad(1);
            e.last = bad(end);
        end
        masks(end + 1) = e; %#ok<AGROW>
    end
    out.masks = masks;

    % The spike consumer: blank its channel, then step1_bandpass.
    label = char(S.spikeLabel);
    col = find(strcmp(cellstr(S.channelLabels), label), 1);
    x = double(S.y(:, col));
    sp = double(M.(['blank_spikes_' label]));
    for k = 1:size(sp, 1)
        x(sp(k, 1):sp(k, 2)) = NaN;
    end
    D = struct();
    D.fs = fs;
    D.y = x;
    D.t = (0:N-1)' / fs;
    D.neuralChannels = 1;
    D.channelLabels = {label};
    P = pipeline_params();
    D = step1_bandpass(D, P, false);
    f = D.filtered(:, 1);
    out.step1_valid = nnz(~isnan(f));
    out.step1_nan = nnz(isnan(f));
    out.step1_nan_equals_blank = isequal(isnan(f), isnan(x));
    out.input_nan_first = find(isnan(x), 1, 'first');
    out.input_nan_last = find(isnan(x), 1, 'last');
    out.max_zero_run_filtered = max_zero_run(f);
    out.max_zero_run_input = max_zero_run(x);
    out.band = [D.bandInfo.low D.bandInfo.high];
    out.step1_from = where;

    fid = fopen(outFile, 'w', 'n', 'UTF-8');
    fwrite(fid, jsonencode(out), 'char');
    fclose(fid);
end

function r = max_zero_run(v)
% Longest run of exact zeros (NaN is not zero).
    z = (v == 0);
    r = 0; c = 0;
    for i = 1:numel(z)
        if z(i)
            c = c + 1;
            r = max(r, c);
        else
            c = 0;
        end
    end
end
