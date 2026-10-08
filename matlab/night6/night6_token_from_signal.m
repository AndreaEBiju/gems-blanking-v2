function token = night6_token_from_signal(sig)
% NIGHT6_TOKEN_FROM_SIGNAL  MATLAB-variable token of a signal name (invariant 22).
%
%   night6_token_from_signal('LVN2-RVN2')   % -> 'LVN2_minus_RVN2'
%
% Twin of gems_blanking_v2.emit.handoff.matlab_signal_token, refusals included: a
% name that already contains '_minus_', has more than one '-', or whose token would
% not read back ('d_minus-fa' -> 'd_minus_minus_fa' -> 'd-minus_fa') has no
% unambiguous token. Used here to cross-check names; the
% mask file's names come from Python.
    sig = char(sig);
    token = strrep(sig, '-', '_minus_');
    bad = contains(sig, '_minus_') || sum(sig == '-') > 1 ...
          || numel(regexp(token, '_minus_', 'start')) > 1 ...
          || ~strcmp(regexprep(token, '_minus_', '-'), sig);
    if bad
        error('night6:badSignal', 'signal name ''%s'' has no unambiguous MATLAB form', sig);
    end
end
