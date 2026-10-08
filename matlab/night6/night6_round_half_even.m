function r = night6_round_half_even(v)
% NIGHT6_ROUND_HALF_EVEN  Python's round() on a double: ties go to the even integer.
%
% The epoch's first sample is round(epochStart_s * fs) on the Python side
% (extent.grid.seconds_to_sample; the Night 4 runner), and Python's round is
% half-to-even while MATLAB's round is half-away-from-zero. They differ only on an
% exact .5 product - rare at fs = 24414.0625, but a one-sample shift of every mask
% when it happens (invariants 15, 22). The product itself is the same IEEE double
% multiplication on both sides.
    r = round(v);
    tie = abs(v - fix(v)) == 0.5;
    r(tie) = 2 * round(v(tie) / 2);
end
