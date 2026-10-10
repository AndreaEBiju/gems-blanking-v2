function k = night6_first_sample(t, fs)
% NIGHT6_FIRST_SAMPLE  First 0-based sample at or after t seconds: ceil(t fs).
%
%   k = night6_first_sample(t, fs)
%
% The MATLAB twin of Python's gems_blanking_v2.extent.grid.first_sample_at_or_after - the
% one conversion of an EVENT time (a stim end, a settling point, the input mask point) to
% the first sample it no longer covers (RULING 2026-10-09 (i) 3: k = ceil(t fs)). A product
% within 1e-6 samples of an integer IS that integer (grid.SAMPLE_TOLERANCE: float noise of
% k / fs * fs), so night6_first_sample(k / fs, fs) == k. Scalar t and fs.
    x = double(t) * double(fs);
    k = round(x);
    if abs(x - k) > 1e-6
        k = ceil(x);
    end
end
