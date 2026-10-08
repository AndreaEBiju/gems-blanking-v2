function sig = night6_signal_from_token(token)
% NIGHT6_SIGNAL_FROM_TOKEN  Signal name from its MATLAB-variable token (invariant 22).
%
%   sig = night6_signal_from_token('LVN2_minus_RVN2')   % -> 'LVN2-RVN2'
%
% The exact inverse of gems_blanking_v2.emit.handoff.matlab_signal_token, and the
% MATLAB twin of its Python inverse signal_from_matlab_token. Same refusals: a token
% holding '-' or more than one '_minus_' is not a token. Counting and replacing are
% NON-OVERLAPPING, left to right, as Python's str.count / str.replace are - strfind
% and strrep count overlapping matches ('_minus_minus_' is one match in Python, two
% for strfind), so regexp / regexprep are used instead. tests/test_night6_wrapper.py
% round-trips names through both languages.
    token = char(token);
    if numel(regexp(token, '_minus_', 'start')) > 1 || any(token == '-')
        error('night6:badToken', '''%s'' is not a MATLAB signal token', token);
    end
    sig = regexprep(token, '_minus_', '-');
end
