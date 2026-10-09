function [Y, fsd, spd] = night6_decimate_masked(X, fs, factors, spans)
% NIGHT6_DECIMATE_MASKED  The decimation check's masked decimation, exactly.
%
%   [Y, fsd, spd] = night6_decimate_masked(X, fs, factors, spans)
%
% RULING 2026-10-09 (c) 6 option (a). Night 6's 'decimated78' slow-wave rate must be
% exactly what the decimation check measured (decim_blank.m / decim_check.m, functions
% decimate_masked and map_spans, reproduced line for line):
%   per stage r in factors (78 = [13 6]):
%     bad = isnan(Y);  F = Y with each column fillmissing 'linear', 'EndValues',
%     'nearest' (TEMPORARY, for the filter only);  F = filtfilt(fir1(20 r, 0.8 / r), 1, F)
%     (zero-phase FIR);  keep rows 1 : r : n r, n = floor(rows / r);  a kept sample is NaN
%     if ANY of its r source samples was NaN (the fill is reverted: invariant 8)
%   fsd = fs / prod(factors).
% Composed, decimated row j (1-based) IS source row (j - 1) R + 1, R = prod(factors), and
% is NaN iff any source row (j - 1) R + 1 .. j R was NaN; rows past floor(N / R) R are
% dropped.
%
% spans (1-based inclusive source rows, [nSeg x 2]) are mapped by the same any-source
% rule: [a, b] -> [floor((a - 1) / R) + 1, floor((b - 1) / R) + 1], clipped to the
% decimated length, a span wholly in the dropped tail removed. The cover of spd equals
% the decimated NaN of every column - asserted here ('night6:decimatedMask'), not assumed,
% when spans are given. The factor must divide 24414 (night6_check_decimation).
    night6_check_decimation(factors, fs);
    Y = X; fsd = fs;
    for r = factors(:)'
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
    spd = zeros(0, 2);
    if nargin < 4, return, end
    R = prod([1, factors(:)']);
    Nd = size(Y, 1);
    if ~isempty(spans)
        spd = [floor((spans(:, 1) - 1) / R) + 1, floor((spans(:, 2) - 1) / R) + 1];
        spd = spd(spd(:, 1) <= Nd, :);
        spd(:, 2) = min(spd(:, 2), Nd);
    end
    cover = false(Nd, 1);
    for j = 1:size(spd, 1), cover(spd(j, 1):spd(j, 2)) = true; end
    if ~isequal(isnan(Y), repmat(cover, 1, size(Y, 2)))
        error('night6:decimatedMask', ['the decimated NaN (%d rows) is not the mapped spans'' ' ...
              'cover (%d rows)'], nnz(any(isnan(Y), 2)), nnz(cover));
    end
end
