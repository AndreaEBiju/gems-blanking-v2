function check_hr_resume(resultFile)
% CHECK_HR_RESUME  Review 7 finding 5: night6_hr_runs_resumable on decoded night6 records.
%
% Each case is a night6_record.json fragment, decoded with jsondecode exactly as
% night6_run_recording's resume step decodes the record; the result file maps the case name
% to "true", or to "false: <why>".
    hrv = night6_hr_outputs({'hrv'});
    br = night6_hr_outputs({'breathing'});
    spk = struct('call', 'process_dataset_v2', 'consumers', {{'spikes'}}, ...
                 'signals', {{'L_T', 'R_T'}});
    runHrv = struct('call', 'HR_BR_HRVAnalysis_beats', 'consumers', {{'hrv'}}, ...
                    'signals', {{'R_T'}}, 'outputs_used', hrv);
    runBr = struct('call', 'HR_BR_HRVAnalysis_beats', 'consumers', {{'breathing'}}, ...
                   'signals', {{'R_T'}}, 'outputs_used', br);
    shared = struct('call', 'HR_BR_HRVAnalysis_beats', 'consumers', {{'hrv', 'breathing'}}, ...
                    'signals', {{'R_T'}});
    noUsed = rmfield(runHrv, 'outputs_used');
    wrongUsed = runHrv;
    wrongUsed.outputs_used = br;
    cases = struct();
    % written as night6_run_recording writes R.runs (a cell of structs), then decoded
    cases.two_calls = roundtrip({spk, runHrv, runBr});
    cases.one_shared_call = roundtrip({spk, shared});
    cases.no_outputs_used = roundtrip({spk, noUsed, runBr});
    cases.outputs_used_of_the_other_consumer = roundtrip({spk, wrongUsed, runBr});
    cases.no_hr_run = roundtrip({spk});
    cases.empty_runs = roundtrip({});
    cases.hr_only_same_fields = roundtrip({runHrv, runBr});   % decodes to a struct array
    cases.no_runs_field = jsondecode('{"status": "complete"}');
    cases.consumer_as_text = jsondecode(['{"runs": [{"call": "HR_BR_HRVAnalysis_beats", ' ...
        '"consumers": "breathing", "outputs_used": ' jsonencode(br) '}]}']);
    cases.hr_without_consumers = jsondecode('{"runs": [{"call": "HR_BR_HRVAnalysis_beats"}]}');
    out = struct();
    for name = fieldnames(cases)'
        [tf, why] = night6_hr_runs_resumable(cases.(name{1}));
        if tf
            out.(name{1}) = 'true';
        else
            out.(name{1}) = ['false: ' why];
        end
    end
    fid = fopen(resultFile, 'w', 'n', 'UTF-8');
    fwrite(fid, jsonencode(out), 'char');
    fclose(fid);
end

function R = roundtrip(runs)
    R = struct('status', 'complete');
    R.runs = runs;
    R = jsondecode(jsonencode(R));
end
