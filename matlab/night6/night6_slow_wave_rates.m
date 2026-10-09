function [rate, R] = night6_slow_wave_rates(name, fs)
% NIGHT6_SLOW_WAVE_RATES  The declared slow-wave sample rates. A REQUIRED declaration.
%
%   rate = night6_slow_wave_rates()            the table: names and stage factors
%   [~, R] = night6_slow_wave_rates(name)      the named mode, checked; refused by name
%   [~, R] = night6_slow_wave_rates(name, fs)  ... and checked against the epoch's rate
%
% RULING 2026-10-09 (c) 6. Every Night 6 batch and every night6_run_recording call must
% NAME the rate slowWaveAnalysis_new runs at - there is no default, because whether
% Night 6 runs decimated is decided by the re-run of the decimation check, not here:
%   'full'         her function on the epoch at its own rate (24414.0625 Hz)
%   'decimated78'  decimated by 78 = 13 x 6 first (night6_decimate_masked: the decimation
%                  check's own zero-phase FIR stages and any-source mask rule)
% R = struct('name', 'factors' (stage factors, [] for full), 'factor' (their product)).
%
% Refused by name:
%   night6:slowWaveRate      no rate declared, or an unknown one
%   night6:decimationFactor  a factor that does not divide 24414 (ruling (c) 6: "Any
%                            decimation factor must divide 24414"), checked on the table
%                            itself, and, with fs, one that does not divide floor(fs)
    rate = struct('name', {'full', 'decimated78'}, 'factors', {[], [13 6]});
    for k = 1:numel(rate)
        night6_check_decimation(rate(k).factors, []);
    end
    if nargin < 1
        R = [];
        return
    end
    if isstring(name), name = char(name); end
    if isempty(name) || ~ischar(name)
        error('night6:slowWaveRate', ['no slow-wave rate declared: name one of [%s] ' ...
              '(RULING 2026-10-09 (c) 6; no default)'], strjoin({rate.name}, ', '));
    end
    hit = rate(strcmp({rate.name}, name));
    if isempty(hit)
        error('night6:slowWaveRate', 'unknown slow-wave rate ''%s''; declared: [%s]', name, ...
              strjoin({rate.name}, ', '));
    end
    R = struct('name', hit.name, 'factors', hit.factors, 'factor', prod([1, hit.factors]));
    if nargin >= 2 && ~isempty(fs)
        night6_check_decimation(hit.factors, fs);
    end
end
