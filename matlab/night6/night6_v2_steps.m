function steps = night6_v2_steps()
% NIGHT6_V2_STEPS  The step functions process_dataset_v2 calls, in order. Declared once.
%
% RULING 2026-10-08 (i) 1 names the list: step1_bandpass, step2_noise_sigma,
% step3_detect, step3b_envelope, step4_waveforms, step5c_modality_test,
% step6_spike_report - her step functions, unchanged, in her order, leaving out
% step1b_remove_cardiac. Heartbeats reach the spike consumer ONLY as the pipeline's
% peri-R NaN spans (the spike consumer's own mask).
%
% Difference from her code, reported to Andrea (build to the ruling until she rules):
% her process_dataset.m (processing_new f93e250, lines 16-23) never calls step1b; its
% cardiac step is step1a_blank_cardiac (line 16), which NaN-blanks +/- 15 ms around
% every R-peak before the filter. The ruling's list leaves step1a out, so it is not
% called here. tests/test_night6_wrapper.py checks this list against the ruling's and
% against her process_dataset.m minus step1a and step1b.
    steps = {'step1_bandpass', 'step2_noise_sigma', 'step3_detect', 'step3b_envelope', ...
             'step4_waveforms', 'step5c_modality_test', 'step6_spike_report'};
end
