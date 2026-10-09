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
% then NaN at every span of the run's consumer's mask - and only that consumer's
% (invariant 2). Masked samples are NaN, never 0 (invariant 1): Andrea's functions
% test isnan. The spans are 1-based inclusive into the epoch, so span [a b] is X(a:b)
% here and file samples i0+a .. i0+b.
%
% Which mask: each column gets the mask of its own signal, except in a run with a
% maskSignal (slow_wave, one ANT channel at a time - Andrea, 2026-10-09), where EVERY
% column gets the mask of run.maskSignal, so slowWaveAnalysis_new's joint any(isnan)
% mask is exactly that channel's mask - asserted ('night6:jointMask'), not assumed.
    rows = plan.i0 + (1:plan.n);
    lead = run.consumers{1};
    X = zeros(plan.n, numel(run.signals));
    for j = 1:numel(run.signals)
        sig = run.signals{j};
        r = plan.recipes.(matlab.lang.makeValidName(sig));
        X(:, j) = (double(Y(rows, r.cols)) * r.weights) * plan.scaleToVolts;
        maskSig = sig;
        if isfield(run, 'maskSignal') && ~isempty(run.maskSignal)
            maskSig = run.maskSignal;
        end
        m = plan.masks(strcmp({plan.masks.consumer}, lead) & strcmp({plan.masks.signal}, maskSig));
        if numel(m) ~= 1
            error('night6:mask', 'expected one mask for %s/%s, found %d', lead, maskSig, numel(m));
        end
        for k = 1:size(m.spans, 1)
            X(m.spans(k, 1):m.spans(k, 2), j) = NaN;
        end
    end
    if isfield(run, 'maskSignal') && ~isempty(run.maskSignal)
        % Asserted, not assumed: slowWaveAnalysis_new's joint any(isnan) mask must be
        % exactly mask i - a NaN anywhere else (in the recording itself, or from a
        % column built otherwise) would silently widen it for the kept channels.
        mask_i = false(plan.n, 1);
        for k = 1:size(m.spans, 1)
            mask_i(m.spans(k, 1):m.spans(k, 2)) = true;
        end
        joint = any(isnan(X), 2);
        if ~isequal(joint, mask_i)
            error('night6:jointMask', ['slow_wave run on %s: the joint NaN mask differs ' ...
                  'from %s''s mask at %d sample(s) (first at epoch row %d)'], ...
                  run.maskSignal, run.maskSignal, nnz(joint ~= mask_i), ...
                  find(joint ~= mask_i, 1));
        end
    end
end
