function W = night6_slow_wave_settings()
% NIGHT6_SLOW_WAVE_SETTINGS  The slowWaveAnalysis_new settings Night 6 runs with. ONE site.
%
%   W = night6_slow_wave_settings()
%
% RULING 2026-10-09 (h) 1: Night 6's slow-wave rate and timing keep batch_process.m's P.sw_*
% (= T / tolerance_sweep) - low-pass on, 0.15 Hz, order 2, 5 s smoothing, 15 s edge buffer;
% NOT run_continuous.m (lowPassOn false, 2 Hz, order 4, window 10, buffer 3). The record's
% params().slow_wave, the arguments night6_call_slow_wave hands her function and the caveat
% text in every slow-wave output's provenance (night6_slow_wave_caveats) are all built from
% this one struct (invariant 33), so they cannot disagree.
    W = struct('lowPassOn', true, 'lowPassCutoff', 0.15, 'lowPassOrder', 2, ...
               'smoothWindow', 5, 'edgeBufferSec', 15, ...
               'blankIdx', ['each run''s masked spans (night6_call_slow_wave; ' ...
                            'RULING 2026-10-09 (c) 6)'], ...
               'source', ['batch_process.m P.sw_* (= T / tolerance_sweep); NOT ' ...
                          'run_continuous.m (lowPassOn false, 2 Hz, order 4, window 10, buffer 3)']);
end
