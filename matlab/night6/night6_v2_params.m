function P = night6_v2_params()
% NIGHT6_V2_PARAMS  The parameter struct process_dataset_v2 hands her steps. Declared once.
%
%   P = night6_v2_params()
%
% Andrea, 2026-10-09: P = pipeline_params(); P.bandpassLow = 300; P.bandpassHigh = 3000;
% everything else her default. RULING 2026-10-08 (k) 3: P.envCardiacGuardMs = 0 - the
% constant peri-R NaN spans of the spike mask already remove those samples from step3b's
% activity RMS (through validMask), so the RMS and the spikes exclude the same samples
% around each beat; her 15 ms guard would cut a different window. (With a guard of 0 her
% step3b still leaves out the R sample itself, which lies inside the span.) The values the
% ruling relies on are ASSERTED here, not assumed: threshSigma 4.5 comes from her
% pipeline_params, so a change to her default would otherwise change the spike consumer
% silently. Refused by name ('process_dataset_v2:params') when threshSigma ~= 4.5,
% bandpassLow ~= 300, bandpassHigh ~= 3000, envCardiacGuardMs ~= 0 or edgeBufferMs ~= 10.5.
% RULING 2026-10-09 (c) 1: P.edgeBufferMs = 10.5 ms - step2's pad around every invalid
% sample, which covers step1's measured zero-phase edge settling (7.782 ms) plus the
% 2.5 ms step4 reads further (10.282 ms, rounded up to 0.5 ms; invariant 19). Her default
% is 10. The value is read from the one declaration (night6_edge_settling:
% edge_settling.json, a copy of gems_blanking_v2.extent.tolerance.EDGE_SETTLING), never
% typed here; the assertion below only checks it. What was actually used is recorded from
% the returned struct (night6_run_recording: runs{}.spike_params, with the declaration's
% hash and the measurement files' SHA-256), never as literal text.
    P = pipeline_params();
    P.bandpassLow = 300;
    P.bandpassHigh = 3000;
    P.envCardiacGuardMs = 0;   % RULING 2026-10-08 (k) 3
    E = night6_edge_settling();
    P.edgeBufferMs = E.spikes.edge_buffer_ms;   % RULING 2026-10-09 (c) 1
    want = struct('threshSigma', 4.5, 'bandpassLow', 300, 'bandpassHigh', 3000, ...
                  'envCardiacGuardMs', 0, 'edgeBufferMs', 10.5);
    for f = fieldnames(want)'
        if ~isfield(P, f{1}) || ~isnumeric(P.(f{1})) || ~isscalar(P.(f{1})) ...
                || P.(f{1}) ~= want.(f{1})
            got = 'absent';
            if isfield(P, f{1}), got = mat2str(P.(f{1})); end
            error('process_dataset_v2:params', ['P.%s is %s, not %g (Andrea 2026-10-09, ' ...
                  'RULING 2026-10-08 (k) 3, 2026-10-09 (c) 1; pipeline_params at %s)'], ...
                  f{1}, got, ...
                  want.(f{1}), which('pipeline_params'));
        end
    end
end
