function E = night6_slow_wave_extent(prov, W)
% NIGHT6_SLOW_WAVE_EXTENT  The caveat every slow-wave result carries, checked against the mask.
%
%   E = night6_slow_wave_extent(prov, W)
%
%   prov  the mask file's provenance (jsondecode of provenance_json)
%   W     night6_slow_wave_settings(): W.extent (candidate C1, floor_z 7.42, band 0-2)
%
% Andrea, 2026-10-12 (go): every slow-wave result says "slow wave is C1 at floor 7.42 and NOT
% covered by the harm test". The sentence is only true of a mask made that way, so where the
% mask provenance records it, it is checked, and a mismatch is refused by name
% ('night6:slowWaveExtent') before anything runs:
%   thresholds.tolerances.extent_rule.consumers.slow_wave  - candidate must be W's; declared
%       false is accepted only for a pilot mask (extra.SW_PILOT), whose C1 spans were added
%       by the pilot from candidate set v2;
%   thresholds.tolerances.cross_band_floor.<band>          - must equal floor_z (to 0.005).
% Absent fields are recorded as unchecked, never assumed.
    X = W.extent;
    E = struct('statement', X.statement, 'candidate', X.candidate, 'floor_z', X.floor_z, ...
               'floor_band', X.floor_band, 'candidate_checked', 'unchecked: no extent_rule for slow_wave in the mask provenance', ...
               'floor_checked', 'unchecked: no cross_band_floor for the band in the mask provenance');
    tol = struct();
    if isfield(prov, 'thresholds') && isstruct(prov.thresholds) ...
            && isfield(prov.thresholds, 'tolerances') && isstruct(prov.thresholds.tolerances)
        tol = prov.thresholds.tolerances;
    end
    if isfield(tol, 'extent_rule') && isstruct(tol.extent_rule) ...
            && isfield(tol.extent_rule, 'consumers') && isfield(tol.extent_rule.consumers, 'slow_wave')
        sw = tol.extent_rule.consumers.slow_wave;
        if isfield(sw, 'declared') && isequal(sw.declared, false)
            if isfield(prov, 'extra') && isstruct(prov.extra) && isfield(prov.extra, 'SW_PILOT')
                E.candidate_checked = ['pilot mask: extent_rule declares slow wave false; ' ...
                                       'extra.SW_PILOT added candidate-set v2 C1 spans'];
            else
                error('night6:slowWaveExtent', ['the mask provenance declares no slow-wave ' ...
                      'extent, yet slow wave runs on it: "%s" would not be true'], X.statement);
            end
        elseif ~isfield(sw, 'candidate') || ~strcmp(char(sw.candidate), X.candidate)
            got = '(none)';
            if isfield(sw, 'candidate'), got = char(sw.candidate); end
            error('night6:slowWaveExtent', ['the mask provenance declares slow-wave extent %s, ' ...
                  'not %s: "%s" would not be true'], got, X.candidate, X.statement);
        else
            E.candidate_checked = sprintf('mask provenance extent_rule: slow_wave candidate %s', ...
                                          X.candidate);
        end
    end
    band = matlab.lang.makeValidName(X.floor_band);   % jsondecode's name for "0-2"
    if isfield(tol, 'cross_band_floor') && isstruct(tol.cross_band_floor) ...
            && isfield(tol.cross_band_floor, band)
        v = double(tol.cross_band_floor.(band));
        if ~(abs(v - X.floor_z) < 0.005)
            error('night6:slowWaveExtent', ['the mask provenance''s %s floor is %g, not %g: ' ...
                  '"%s" would not be true'], X.floor_band, v, X.floor_z, X.statement);
        end
        E.floor_checked = sprintf('mask provenance cross_band_floor %s = %g', X.floor_band, v);
    end
end
