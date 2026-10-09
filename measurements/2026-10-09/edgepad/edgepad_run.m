function edgepad_run()
% EDGEPAD_RUN  Push edge test inputs through Andrea's step1_bandpass (v2 params), save outputs.
% Read only: processing_new (GEMS_PROCESSING_NEW) and the wrapper's night6_v2_params.
% Writes only edgepad/out_*.mat beside this file.
    here = fileparts(mfilename('fullpath'));
    pn = getenv('GEMS_PROCESSING_NEW');
    assert(~isempty(pn) && isfolder(pn), 'GEMS_PROCESSING_NEW not set');
    addpath(pn);
    addpath(fullfile('C:\Users\User\Documents\gems-blanking-v2-instructions\.claude\worktrees', ...
        'agent-a0136abf4ea65b4dc', 'matlab', 'night6'));
    P = night6_v2_params();
    fs = 24414.0625;
    info = struct('step1', which('step1_bandpass'), 'params', which('night6_v2_params'), ...
        'pipeline_params', which('pipeline_params'), 'fs', fs, ...
        'bandpassLow', P.bandpassLow, 'bandpassHigh', P.bandpassHigh, ...
        'filterOrder', P.filterOrder, 'edgeBufferMs', P.edgeBufferMs, 'version', version);

    % ---------------- synthetic deterministic edges: 3 s columns, one gap at 1.5 s
    N = round(3 * fs);
    c0 = round(1.5 * fs);                 % first NaN sample (1-based)
    Gms = [1 21 100 1000];
    names = {}; X = []; XT = []; GAP = [];
    t = ((1:N)' - 1) / fs;
    for G = Gms
        gN = round(G * 1e-3 * fs);
        gap = false(N, 1); gap(c0:c0 + gN - 1) = true;
        lt = c0 - 1; ld = c0 + gN;        % last valid before, first valid after
        cols = {};
        x = zeros(N, 1); x(ld:end) = 1;                     cols(end+1, :) = {'step', x};
        x = zeros(N, 1); x(lt) = 1;                         cols(end+1, :) = {'spike_trail', x};
        x = zeros(N, 1); x(ld) = 1;                         cols(end+1, :) = {'spike_lead', x};
        sg = 0.5e-3 * fs;                                   % 0.5 ms Gaussian bump on the edge
        x = exp(-0.5 * (((1:N)' - c0) / sg).^2);            cols(end+1, :) = {'bump_trail', x};
        x = exp(-0.5 * (((1:N)' - (ld - 1)) / sg).^2);      cols(end+1, :) = {'bump_lead', x};
        n1 = round(1e-3 * fs);                              % one 1 kHz cycle touching the edge
        x = zeros(N, 1); x(lt - n1 + 1:lt) = sin(2 * pi * (0:n1 - 1)' / n1); cols(end+1, :) = {'biph_trail', x};
        x = zeros(N, 1); x(ld:ld + n1 - 1) = sin(2 * pi * (0:n1 - 1)' / n1); cols(end+1, :) = {'biph_lead', x};
        for f = [1 10 50 150]                               % sub-band sinusoid: the fill's kinks
            x = sin(2 * pi * f * (t - t(c0)));              % zero crossing (max slope) at the edge
            cols(end+1, :) = {sprintf('kink_%dHz', f), x}; %#ok<AGROW>
        end
        for k = 1:size(cols, 1)
            names{end+1} = sprintf('%s_G%d', cols{k, 1}, G); %#ok<AGROW>
            XT(:, end+1) = cols{k, 2}; %#ok<AGROW>
            GAP(:, end+1) = gap; %#ok<AGROW>
        end
    end
    [Y, Yref, Yraw] = run_step1(XT, logical(GAP), fs, P);
    save(fullfile(here, 'out_syn.mat'), 'names', 'XT', 'GAP', 'Y', 'Yref', 'Yraw', 'c0', 'Gms', 'info', '-v7');
    clear XT GAP Y Yref Yraw

    % ---------------- white noise: 60 s columns, many gaps
    rng(20261009);
    Nn = round(60 * fs);
    spec = [21 250; 1 250; 100 500; 1000 2000];   % [gap ms, period ms]
    XT = repmat(randn(Nn, 1), 1, size(spec, 1));  % same noise, different gaps
    GAP = false(Nn, size(spec, 1));
    for j = 1:size(spec, 1)
        gN = round(spec(j, 1) * 1e-3 * fs); per = round(spec(j, 2) * 1e-3 * fs);
        for s = round(0.5 * fs):per:Nn - round(0.5 * fs)
            GAP(s:s + gN - 1, j) = true;
        end
    end
    [Y, Yref, Yraw] = run_step1(XT, GAP, fs, P);
    save(fullfile(here, 'out_noise.mat'), 'GAP', 'Y', 'Yref', 'spec', 'info', '-v7');
    clear XT GAP Y Yref Yraw

    % ---------------- real tripole snippets: peri-R spans at the routed beats, and 100 ms / 1 s gaps
    S = load(fullfile(here, 'snippets.mat'));
    nb = round(11.5e-3 * fs); na = round(9.5e-3 * fs);   % [R - 11.5 ms, R + 9.5 ms)
    XT = []; GAP = false(0, 0); rnames = {};
    for i = 1:3
        x = S.(sprintf('x%d', i)); x = x(:); r = S.(sprintf('r%d', i)); r = r(:);
        Ns = numel(x);
        g = false(Ns, 1);
        for k = 1:numel(r)              % r is 0-based; span 0-based [r-nb, r+na)
            a = r(k) - nb + 1; b = r(k) + na;    % -> 1-based inclusive
            g(max(1, a):min(Ns, b)) = true;
        end
        g(1:round(0.2 * fs)) = false; g(end - round(0.2 * fs):end) = false;  % keep file ends out
        XT(:, end+1) = x; GAP(:, end+1) = g; rnames{end+1} = sprintf('real%d_perir', i); %#ok<AGROW>
        g = false(Ns, 1);
        for s = round(0.5 * fs):round(0.5 * fs):Ns - round(0.5 * fs); g(s:s + round(0.1 * fs) - 1) = true; end
        XT(:, end+1) = x; GAP(:, end+1) = g; rnames{end+1} = sprintf('real%d_g100', i); %#ok<AGROW>
        g = false(Ns, 1);
        for s = round(1 * fs):round(3 * fs):Ns - round(2 * fs); g(s:s + round(1 * fs) - 1) = true; end
        XT(:, end+1) = x; GAP(:, end+1) = g; rnames{end+1} = sprintf('real%d_g1000', i); %#ok<AGROW>
    end
    [Y, Yref, Yraw] = run_step1(XT, GAP, fs, P); %#ok<ASGLU>
    save(fullfile(here, 'out_real.mat'), 'rnames', 'GAP', 'Y', 'Yref', 'XT', 'info', '-v7');
    fprintf('edgepad_run done\n');
end

function [Y, Yref, Yraw] = run_step1(XT, GAP, fs, P)
% Y: step1_bandpass on XT with GAP set to NaN (fill, filtfilt, NaN restore - her code).
% Yref: step1_bandpass on XT with no gap. Yraw: same as Y before the NaN restore, rebuilt
% from her own lines (step1_bandpass.m:30, 52-57) to locate the transient's peak in the gap.
    X = XT; X(GAP) = NaN;
    Y = call_step1(X, fs, P);
    Yref = call_step1(XT, fs, P);
    [b, a] = butter(P.filterOrder, [P.bandpassLow min(P.bandpassHigh, fs/2 - 1)] / (fs/2), 'bandpass');
    Yraw = zeros(size(X));
    for k = 1:size(X, 2)
        xf = fillmissing(X(:, k), 'linear', 'EndValues', 'nearest');
        Yraw(:, k) = filtfilt(b, a, xf);
        v = ~GAP(:, k);
        assert(isequal(Yraw(v, k), Y(v, k)), 'rebuilt path differs from step1_bandpass');
    end
end

function Yf = call_step1(X, fs, P)
    D = struct('fs', fs, 'y', X, 'neuralChannels', 1:size(X, 2));
    D.channelLabels = arrayfun(@(k) sprintf('c%d', k), 1:size(X, 2), 'UniformOutput', false);
    D = evalc_step1(D, P);
    Yf = D.filtered;
end

function D = evalc_step1(D, P)
    [~, D] = evalc('step1_bandpass(D, P, false)');   % silence her per-channel prints
end

function varargout = deal_col(varargin) %#ok<DEFNU>
    varargout = varargin;
end
