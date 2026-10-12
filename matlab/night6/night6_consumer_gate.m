function [status, reason, heldSignals] = night6_consumer_gate(consumer, held, elsewhere, ...
                                                              subset, signals)
% NIGHT6_CONSUMER_GATE  Whether one consumer runs from this mask file - the ONE site.
%
%   [status, reason, heldSignals] = night6_consumer_gate(consumer, held, elsewhere, subset, ...
%                                                        signals)
%
%   consumer   a consumer name ('spikes', 'hrv', ...)
%   held       the decoded heldconsumers_json: a field per HOLD UNIT QC holds in this
%              recording (night6_hold_unit: the consumer; spikes per cuff, spikes_L /
%              spikes_R - RULING 2026-10-10 item 1, Andrea revised 2026-10-10), with its
%              reasons and, when Andrea (or the pilot's stand-in run) released it, 'released'.
%              A file from before the per-cuff unit holds 'spikes' as a whole: every spike
%              signal is held then.
%   elsewhere  the decoded elsewhere_json (RULING 2026-10-10 item 5): a field per consumer
%              this file does not finish, naming the file that does
%   subset     the batch list's consumer subset for this recording; {} runs every consumer
%   signals    the consumer's signals in this file (its masks)
%
% heldSignals lists the signals whose unit is held and not released (a held spike cuff); the
% other signals run - masks are never merged across cuffs. When every signal is held the
% status is 'skipped_held'.
%
% status is '' (no objection - the plan's own checks decide) or, in this order:
%   'skipped_elsewhere'     the consumer is finished in another mask file (its masks are
%                           not in this one)
%   'skipped_not_in_batch'  the batch runs a subset of the consumers for this recording
%   'skipped_held'          QC holds the consumer in this recording and it is not released;
%                           the other consumers run (never the whole recording)
    status = '';
    reason = '';
    heldSignals = {};
    if isfield(elsewhere, consumer)
        status = 'skipped_elsewhere';
        reason = char(elsewhere.(consumer));
        return
    end
    if ~isempty(subset) && ~ismember(consumer, subset)
        status = 'skipped_not_in_batch';
        reason = sprintf(['the batch list runs [%s] for this recording (RULING 2026-10-10 ' ...
                          'item 5)'], strjoin(subset, ' '));
        return
    end
    signals = cellstr(signals);
    if isempty(signals), signals = {''}; end
    r = {};
    for k = 1:numel(signals)
        if strcmp(consumer, 'spikes') && ~isfield(held, 'spikes') && ~isempty(signals{k})
            unit = night6_hold_unit(consumer, signals{k});
        else
            unit = consumer;   % a non-spike consumer, or a whole-consumer hold (older file)
        end
        if ~isfield(held, unit), continue, end
        h = held.(unit);
        released = isstruct(h) && isfield(h, 'released') && ~isempty(strtrim(char(h.released)));
        if released, continue, end
        heldSignals{end + 1} = signals{k}; %#ok<AGROW>
        if isstruct(h) && isfield(h, 'reasons'), r = [r, cellstr(h.reasons)']; end %#ok<AGROW>
    end
    if isempty(heldSignals), return, end
    r = unique(r, 'stable');
    if numel(heldSignals) == numel(signals)
        status = 'skipped_held';
        reason = ['QC holds this consumer in this recording (RULING 2026-10-10 item 1): ' ...
                  strjoin(r, '; ')];
    else
        reason = ['QC holds ' strjoin(heldSignals, ', ') ' of this consumer (per cuff, ' ...
                  'Andrea revised 2026-10-10); the others run: ' strjoin(r, '; ')];
    end
end
