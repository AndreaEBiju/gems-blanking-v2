function [tf, why] = night6_span_rule_resumable(R, subset, ruleNow)
% NIGHT6_SPAN_RULE_RESUMABLE  The slow-wave span rule's part of the resume key.
%
%   [tf, why] = night6_span_rule_resumable(R, subset, ruleNow)
%
%   R        a complete night6_record.json, decoded (jsondecode)
%   subset   this batch's consumer subset (night6_run_recording 'Consumers'; {} = all)
%   ruleNow  night6_slow_wave_settings().spanRule.name
%
% RULING 2026-10-12: an epoch whose slow-wave outputs were made under another span rule (or
% before there was one) must rerun - but ONLY when slow wave is in this batch's consumers.
% Review of 3ce6575: requiring R.slow_wave_span_rule unconditionally made every existing
% {hrv, breathing, mmc} record (written by f844719, which has no such field) rerun and have
% its outputs moved aside. So:
%   slow_wave not in the subset (a non-empty subset without it)       -> resumable;
%   every consumer ({}), and the record's slow wave did not run
%     (status skipped_* or reads_nothing)                             -> resumable;
%   otherwise the record must carry slow_wave_span_rule == ruleNow.
    if isempty(subset), subset = {}; else, subset = cellstr(subset); end
    if ~isempty(subset) && ~any(strcmp(subset, 'slow_wave'))
        tf = true;
        why = 'slow_wave is not in this batch''s consumers';
        return
    end
    if isempty(subset) && isfield(R, 'consumers') && isstruct(R.consumers) ...
            && isfield(R.consumers, 'slow_wave') && isfield(R.consumers.slow_wave, 'status')
        st = char(R.consumers.slow_wave.status);
        if startsWith(st, 'skipped') || strcmp(st, 'reads_nothing')
            tf = true;
            why = sprintf('slow wave did not run in this record (%s)', st);
            return
        end
    end
    if ~isfield(R, 'slow_wave_span_rule')
        tf = false;
        why = 'its slow-wave outputs record no span rule (made before RULING 2026-10-12)';
        return
    end
    if ~strcmp(char(R.slow_wave_span_rule), ruleNow)
        tf = false;
        why = sprintf('its slow-wave outputs were made under span rule %s, not %s', ...
                      char(R.slow_wave_span_rule), ruleNow);
        return
    end
    tf = true;
    why = '';
end
