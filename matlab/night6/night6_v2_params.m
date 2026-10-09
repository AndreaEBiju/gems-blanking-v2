function P = night6_v2_params()
% NIGHT6_V2_PARAMS  The parameter struct process_dataset_v2 hands her steps. Declared once.
%
%   P = night6_v2_params()
%
% Andrea, 2026-10-09: P = pipeline_params(); P.bandpassLow = 300; P.bandpassHigh = 3000;
% everything else her default. The values the ruling relies on are ASSERTED here, not
% assumed: threshSigma 4.5 comes from her pipeline_params, so a change to her default
% would otherwise change the spike consumer silently. Refused by name
% ('process_dataset_v2:params') when threshSigma ~= 4.5, bandpassLow ~= 300 or
% bandpassHigh ~= 3000. What was actually used is recorded from the returned struct
% (night6_run_recording: runs{}.spike_params), never as literal text.
    P = pipeline_params();
    P.bandpassLow = 300;
    P.bandpassHigh = 3000;
    want = struct('threshSigma', 4.5, 'bandpassLow', 300, 'bandpassHigh', 3000);
    for f = fieldnames(want)'
        if ~isfield(P, f{1}) || ~isnumeric(P.(f{1})) || ~isscalar(P.(f{1})) ...
                || P.(f{1}) ~= want.(f{1})
            got = 'absent';
            if isfield(P, f{1}), got = mat2str(P.(f{1})); end
            error('process_dataset_v2:params', ['P.%s is %s, not %g (Andrea 2026-10-09; ' ...
                  'pipeline_params at %s)'], f{1}, got, want.(f{1}), which('pipeline_params'));
        end
    end
end
