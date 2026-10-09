function kept = night6_keep_slow_wave(outDir, label, base, signals, keep, maskSignal, newFiles)
% NIGHT6_KEEP_SLOW_WAVE  Keep only the analysed channels' outputs of one slow-wave run.
%
%   kept = night6_keep_slow_wave(outDir, label, base, signals, keep, maskSignal, newFiles)
%
% Andrea, 2026-10-09: slow wave runs one ANT channel at a time. The run that applied
% maskSignal's mask to all three columns (signals, her column order) is valid ONLY for
% the channels in keep - maskSignal itself, plus any channel whose own mask is
% identical. slowWaveAnalysis_new (called unchanged) saved <label>_slowWaves.mat for
% all three columns; this splits out each kept channel as
% <base>_slowWaves_<channel>.mat and DELETES the three-channel file, so no column
% analysed under another channel's mask survives to be read by mistake.
%
% Per-channel variables are that channel's column (slowWaveTimeSeries,
% slowWaveRateSeries, avgSlowWave, slowWavePeakLocs, sw_implausibleFraction); every
% other variable of her file (invalidMask, edgeMask, t, fs, window, ...) is shared and
% copied as is. Added: channel, channelColumn, maskSignal, keptFrom.
%
% Her diagnostic figures (Figures on) plot column 1 only: they are kept when column 1
% is a kept channel and deleted otherwise. newFiles is every file the call created.
    swName = sprintf('%s_slowWaves.mat', label);
    swFile = fullfile(outDir, swName);
    if ~isfile(swFile)
        error('night6:slowWaveOutput', 'slowWaveAnalysis_new wrote no %s', swName);
    end
    S = load(swFile);
    perChannel = {'slowWaveTimeSeries', 'slowWaveRateSeries', 'avgSlowWave', ...
                  'slowWavePeakLocs', 'sw_implausibleFraction'};
    shared = setdiff(fieldnames(S), perChannel, 'stable');
    kept = struct('mask_signal', maskSignal, 'channels', {keep}, 'files', {{}}, ...
                  'deleted', {{}});
    for s = keep(:)'
        ci = find(strcmp(signals, s{1}));
        if numel(ci) ~= 1
            error('night6:slowWaveKeep', 'kept channel %s is not one of [%s]', s{1}, ...
                  strjoin(signals, ' '));
        end
        out = struct();
        for f = shared(:)', out.(f{1}) = S.(f{1}); end
        out.slowWaveTimeSeries = S.slowWaveTimeSeries(:, ci);
        out.slowWaveRateSeries = S.slowWaveRateSeries(:, ci);
        out.avgSlowWave = S.avgSlowWave(ci);
        out.slowWavePeakLocs = S.slowWavePeakLocs{ci};
        out.sw_implausibleFraction = S.sw_implausibleFraction(ci);
        out.channel = s{1};
        out.channelColumn = ci;
        out.maskSignal = maskSignal;
        out.keptFrom = swName;
        name = sprintf('%s_slowWaves_%s.mat', base, s{1});
        save(fullfile(outDir, name), '-struct', 'out');
        kept.files{end + 1} = name;
    end
    delete(swFile);
    kept.deleted{end + 1} = swName;
    keepFigures = any(strcmp(keep, signals{1}));
    for f = newFiles(:)'
        if strcmp(f{1}, swName) || keepFigures || ~isfile(fullfile(outDir, f{1}))
            continue
        end
        delete(fullfile(outDir, f{1}));
        kept.deleted{end + 1} = f{1};
    end
end
