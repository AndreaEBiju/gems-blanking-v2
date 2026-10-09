function [D, info] = process_dataset_v2(D, varargin)
% PROCESS_DATASET_V2  Andrea's spike pipeline for Night 6 (RULING 2026-10-08 (i)).
%
%   [D, info] = process_dataset_v2(D)
%   [D, info] = process_dataset_v2(D, 'PlotMode', false, 'NanPadMs', 5)
%
% Ruling (i): Andrea's spike results come from processing_new's process_dataset, not
% from detectSortNerveSpikesECAP. This file lives OUTSIDE processing_new and copies
% nothing of hers: it builds her parameter struct and calls her step functions,
% unchanged, in her order, as ruling (i) 1 lists them (night6_v2_steps - the one list
% both this file and the provenance read): step1_bandpass, step2_noise_sigma,
% step3_detect, step3b_envelope, step4_waveforms, step5c_modality_test,
% step6_spike_report. Neither step1b_remove_cardiac nor her step1a_blank_cardiac is
% called: heartbeats reach the spike consumer only as the peri-R NaN spans of its own
% mask (ruling (i) 1).
%
%   P = night6_v2_params(): pipeline_params(), bandpassLow = 300, bandpassHigh = 3000,
%   everything else her default; threshSigma 4.5 and the band are ASSERTED there
%   ('process_dataset_v2:params'), and info.P is what was used.
%
% D is the struct her headless loader bulk_load_one builds: D.y (samples x channels,
% VOLTS, masked samples NaN), D.fs, D.neuralChannels, D.channelLabels, D.rpeakSamples
% (1-based SAMPLE indices into D.y, possibly empty), D.rpeakTimes, D.removedSegmentIdx.
% D.rpeakSamples must be present: among the steps called, step3b_envelope is its only
% reader (step3b_envelope.m:56-61, the cardiac guard on the activity RMS);
% step3_detect reads D.rpeakTimes only to draw its figure (step3_detect.m:155).
%
% Fallback, RULING 2026-10-08 (j) 1: 'Step1aChannels' (logical, one per neural channel,
% default none) names the channels of an animal x cuff on the declared fallback list
% (night6_step1a_fallback): a peri-R excess classified as leak. On those channels only,
% her step1a_blank_cardiac, unchanged, sets +/-15 ms around each R-peak to NaN before
% step1_bandpass; it is refused when there are no R-peaks. info.step1a_channels records it.
%
% Refused by name before any step (invariant 41): a neural channel with no finite
% sample ('process_dataset_v2:noValidSamples') or whose finite samples are all equal
% ('process_dataset_v2:constantInput'), and R-peaks that are not integer samples in D.y.
%
% NaN (invariant 1): this file fills nothing. Her step1_bandpass fills NaN only to
% filter and restores them in D.filtered (step1_bandpass.m:51-58). After the steps,
% every masked input sample of neural channel k must be NaN in D.filtered(:, k)
% ('process_dataset_v2:nanFilled') and invalid in D.validMask(:, k)
% ('process_dataset_v2:nanValid'), the channel named. D.y is not checked: no step
% writes it, so a check on it could never fail.
%
% Spikes in masked time: VERIFIED, not enforced. After the steps, no accepted spike
% (D.spikes(k).alignedCenters) may lie on a NaN sample of D.y or within NanPadMs of one
% (default 5 ms, ruling (i) 3(c)); otherwise 'process_dataset_v2:spikeInMask' names
% the channel. Her step2 already drops NaN plus a 10 ms pad from detection
% (step2_noise_sigma.m:73-80, step3_detect.m:71) and step4 re-aligns by at most 0.5 ms,
% so a hit means her steps changed underneath this file - removing such spikes quietly
% would hide exactly that.
%
% info: steps (name, path), P, n_rpeaks, and spike_check per channel
% (n_spikes, n_in_pad = 0, pad_samples, min_gap_samples to the nearest NaN, absent
% when the channel has no NaN or no spike).
    ip = inputParser;
    ip.addRequired('D', @isstruct);
    ip.addParameter('PlotMode', false, @(x) islogical(x) || isnumeric(x));
    ip.addParameter('NanPadMs', 5, @(x) isnumeric(x) && isscalar(x) && x >= 0);
    ip.addParameter('Step1aChannels', [], @(x) isempty(x) || islogical(x));
    ip.parse(D, varargin{:});
    plotMode = logical(ip.Results.PlotMode);
    padMs = ip.Results.NanPadMs;

    for f = {'y', 'fs', 'neuralChannels', 'channelLabels', 'rpeakSamples'}
        if ~isfield(D, f{1})
            error('process_dataset_v2:missing', 'D.%s is required (her bulk_load_one struct)', f{1});
        end
    end
    N = size(D.y, 1);
    ch = D.neuralChannels(:)';
    labels = cellstr(D.channelLabels);
    for k = 1:numel(ch)
        x = D.y(:, ch(k));
        fin = x(isfinite(x));
        if isempty(fin)
            error('process_dataset_v2:noValidSamples', ...
                  'channel %s has no finite sample: refused, not analysed', labels{k});
        end
        if all(fin == fin(1))
            error('process_dataset_v2:constantInput', ...
                  'channel %s is constant (%.6g) over all %d finite samples: refused', ...
                  labels{k}, fin(1), numel(fin));
        end
    end
    r = double(D.rpeakSamples(:));
    if any(r ~= fix(r)) || any(r < 1 | r > N)
        error('process_dataset_v2:rpeaks', 'D.rpeakSamples must be integer samples in 1..%d', N);
    end

    P = night6_v2_params();   % asserts threshSigma 4.5 and 300-3000 Hz

    nanIn = isnan(D.y(:, ch));
    steps = night6_v2_steps();
    info = struct();
    info.steps = cellfun(@(s) struct('name', s, 'path', which(s)), steps, 'UniformOutput', false);

    % Ruling (j) 1 fallback: on the declared channels only (a leak-classified animal x
    % cuff, night6_step1a_fallback), her step1a_blank_cardiac - unchanged - is applied
    % first; its NaN windows are taken for those channels and no other.
    fb = ip.Results.Step1aChannels;
    if isempty(fb), fb = false(1, numel(ch)); end
    if numel(fb) ~= numel(ch)
        error('process_dataset_v2:step1aChannels', ...
              'Step1aChannels has %d entries for %d neural channels', numel(fb), numel(ch));
    end
    info.step1a_channels = labels(fb(:)');
    if any(fb)
        if isempty(r)
            error('process_dataset_v2:step1aNoRpeaks', ['step1a fallback requested for ' ...
                  '[%s] but D.rpeakSamples is empty: refused, not run unblanked'], ...
                  strjoin(labels(fb(:)'), ' '));
        end
        B = step1a_blank_cardiac(D, P, plotMode);
        D.y(:, ch(fb)) = B.y(:, ch(fb));
        info.step1a = struct('path', which('step1a_blank_cardiac'), ...
                             'win_ms', B.cardiacBlankWinMs, 'n_rpeaks', numel(r));
        clear B
    end
    for s = steps
        D = feval(s{1}, D, P, plotMode);
    end

    % Invariant 1, on what the steps WROTE: the input's NaN must stay NaN in D.filtered
    % and be invalid in D.validMask, channel by channel.
    for k = 1:numel(ch)
        lost = nanIn(:, k) & ~isnan(D.filtered(:, k));
        if any(lost)
            error('process_dataset_v2:nanFilled', ['channel %s: %d masked input sample(s) ' ...
                  'are not NaN in D.filtered (first at sample %d; invariant 1)'], ...
                  labels{k}, nnz(lost), find(lost, 1));
        end
        lost = nanIn(:, k) & D.validMask(:, k);
        if any(lost)
            error('process_dataset_v2:nanValid', ['channel %s: %d masked input sample(s) ' ...
                  'are valid in D.validMask (first at sample %d; invariant 1)'], ...
                  labels{k}, nnz(lost), find(lost, 1));
        end
    end
    still = isnan(D.y(:, ch));   % the spike check's NaN: the input's, plus any step1a span
    pad = ceil(padMs * 1e-3 * D.fs);
    info.P = P;
    info.n_rpeaks = numel(r);
    info.nan_pad_ms = padMs;
    info.spike_check = cell(1, numel(ch));
    for k = 1:numel(ch)
        bad = still(:, k);
        if pad > 0 && any(bad)
            bad = movmax(double(bad), [pad pad]) > 0;
        end
        c = double(D.spikes(k).alignedCenters(:));
        hit = c(bad(c));
        sc = struct('label', labels{k}, 'n_spikes', numel(c), 'n_in_pad', numel(hit), ...
                    'pad_samples', pad);
        if any(still(:, k)) && ~isempty(c)
            idx = (1:N)';
            prev = idx; prev(~still(:, k)) = 0; prev = cummax(prev);          % last NaN at or before
            nxt = idx; nxt(~still(:, k)) = Inf; nxt = flipud(cummin(flipud(nxt)));  % next NaN at or after
            before = c - prev(c);
            before(prev(c) == 0) = Inf;
            gap = min([before, nxt(c) - c], [], 2);
            sc.min_gap_samples = min(gap);
        end
        info.spike_check{k} = sc;
        if ~isempty(hit)
            error('process_dataset_v2:spikeInMask', ['channel %s: %d spike(s) inside a NaN ' ...
                  'span or its %g ms pad (first at sample %d)'], labels{k}, numel(hit), padMs, hit(1));
        end
    end
end
