function check_consumer_gate(outFile)
% CHECK_CONSUMER_GATE  Cases for night6_consumer_gate and night6_consumer_subset (RULING
% 2026-10-10 items 1 and 5); results as JSON for tests/test_night6_consumer_gate.py.
%
% Records are decoded with jsondecode from the JSON the Python writer emits
% (heldconsumers_json, elsewhere_json), so the fields are what Night 6 reads.
    held = jsondecode(['{"slow_wave": {"reasons": ["slow_wave|ANT1|0-2: 24.0% blanked > 20%"]},' ...
                       ' "mmc": {"reasons": ["mmc|ANT1|2-50: retention 0.7 < 0.8"],' ...
                       ' "released": "Andrea 2026-10-11: keep"}}']);
    away = jsondecode('{"spikes": "finished per animal in masks/m_spikes"}');
    none = struct();
    R = struct();
    R.held_unreleased = gate('slow_wave', held, none, {});
    R.held_released = gate('mmc', held, none, {});
    R.not_held = gate('hrv', held, none, {});
    R.elsewhere = gate('spikes', held, away, {});
    R.elsewhere_beats_batch = gate('spikes', held, away, {'spikes'});
    R.not_in_batch = gate('hrv', none, none, {'mmc', 'slow_wave'});
    R.in_batch_but_held = gate('slow_wave', held, none, {'slow_wave'});
    R.in_batch = gate('slow_wave', none, none, {'slow_wave'});
    R.empty_records = gate('breathing', none, none, {});
    % Andrea, revised 2026-10-10: spikes held per cuff - the other cuff runs
    cuffs = jsondecode(['{"spikes_R": {"reasons": ["spikes|R_T|300-3000: 100.0% blanked > 20%"]},' ...
                        ' "spikes_L": {"reasons": ["spikes|L_T|300-3000: 30.0% blanked > 20%"],' ...
                        ' "released": "Andrea: keep L"}}']);
    R.spike_cuff_held = gate('spikes', cuffs, none, {}, {'L_T', 'R_T'});
    R.spike_both_held = gate('spikes', rmfield(cuffs, 'spikes_L'), none, {}, {'R_T'});
    R.spike_whole_held_old = gate('spikes', jsondecode('{"spikes": {"reasons": ["old"]}}'), ...
                                  none, {}, {'L_T', 'R_T'});
    R.unit_L = night6_hold_unit('spikes', 'L_T');
    R.unit_RV2 = night6_hold_unit('spikes', 'R_V2');
    R.unit_hrv = night6_hold_unit('hrv', 'LVN2_minus_RVN1');
    R.unit_bad = err(@() night6_hold_unit('spikes', 'ECG'));
    known = {'spikes', 'slow_wave', 'mmc', 'hrv', 'breathing', 'velocity'};
    [s, t] = night6_consumer_subset({'slow_wave', 'hrv', 'mmc', 'breathing'}, known);
    R.subset_nonspike = struct('subset', strjoin(s, ','), 'tag', t);
    [s, t] = night6_consumer_subset({'spikes'}, known);
    R.subset_spikes = struct('subset', strjoin(s, ','), 'tag', t);
    [s, t] = night6_consumer_subset({}, known);
    R.subset_all = struct('subset', strjoin(s, ','), 'tag', t);
    R.subset_unknown = err(@() night6_consumer_subset({'spikes', 'heart'}, known));
    R.subset_twice = err(@() night6_consumer_subset({'mmc', 'mmc'}, known));
    fid = fopen(outFile, 'w', 'n', 'UTF-8');
    fwrite(fid, jsonencode(R), 'char');
    fclose(fid);
end

function g = gate(c, held, away, subset, signals)
    if nargin < 5, signals = {'X'}; end
    [st, why, hs] = night6_consumer_gate(c, held, away, subset, signals);
    g = struct('status', st, 'reason', why, 'held_signals', {hs});
end

function e = err(f)
    e = struct('id', '', 'message', '');
    try
        f();
    catch ME
        e = struct('id', ME.identifier, 'message', ME.message);
    end
end
