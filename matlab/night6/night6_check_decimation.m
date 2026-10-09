function night6_check_decimation(factors, fs)
% NIGHT6_CHECK_DECIMATION  Refuse a decimation that does not divide the cohort's rate.
%
%   night6_check_decimation(factors, fs)
%
% RULING 2026-10-09 (c) 6: "Any decimation factor must divide 24414." factors are the
% stage factors (their product is the factor); [] is no decimation. Each stage factor
% must be an integer >= 2, and the product must divide 24414 - the TDT rate's whole
% part, so a decimated sample is a whole number of source samples at a whole-hertz
% grid - and, when fs is given, floor(fs) too. Refused by name
% ('night6:decimationFactor'), never rounded.
    COHORT_HZ = 24414;
    if isempty(factors), return, end
    f = double(factors(:)');
    if any(f < 2 | f ~= round(f))
        error('night6:decimationFactor', ['decimation stage factors %s: each must be an ' ...
              'integer >= 2'], mat2str(f));
    end
    r = prod(f);
    if mod(COHORT_HZ, r) ~= 0
        error('night6:decimationFactor', ['decimation factor %d (stages %s) does not divide ' ...
              '%d (RULING 2026-10-09 (c) 6)'], r, mat2str(f), COHORT_HZ);
    end
    if nargin >= 2 && ~isempty(fs) && mod(floor(fs), r) ~= 0
        error('night6:decimationFactor', ['decimation factor %d does not divide this ' ...
              'epoch''s rate %.6f Hz (floor %d)'], r, fs, floor(fs));
    end
end
