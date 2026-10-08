function X = night6_consumer_input(plan, Y, run)
% NIGHT6_CONSUMER_INPUT  One run's input, in volts, with ITS consumer's mask as NaN.
%
%   X = night6_consumer_input(plan, Y, run)
%
%   Y    the whole recording, nFile x nChannels, in the declared units
%   run  an element of plan.runs
%
% Returns plan.n x numel(run.signals) double: the epoch rows i0+1 .. i0+n of each
% signal's recipe (raw contact, pairs lead, or software tripole), scaled to volts,
% then NaN at every span of blank_<consumer>_<signal> for the run's consumer - and
% only that consumer's (invariant 2). Masked samples are NaN, never 0 (invariant 1):
% Andrea's functions test isnan. The spans are 1-based inclusive into the epoch, so
% span [a b] is X(a:b) here and file samples i0+a .. i0+b.
    rows = plan.i0 + (1:plan.n);
    lead = run.consumers{1};
    X = zeros(plan.n, numel(run.signals));
    for j = 1:numel(run.signals)
        sig = run.signals{j};
        r = plan.recipes.(matlab.lang.makeValidName(sig));
        X(:, j) = (double(Y(rows, r.cols)) * r.weights) * plan.scaleToVolts;
        m = plan.masks(strcmp({plan.masks.consumer}, lead) & strcmp({plan.masks.signal}, sig));
        if numel(m) ~= 1
            error('night6:mask', 'expected one mask for %s/%s, found %d', lead, sig, numel(m));
        end
        for k = 1:size(m.spans, 1)
            X(m.spans(k, 1):m.spans(k, 2), j) = NaN;
        end
    end
end
