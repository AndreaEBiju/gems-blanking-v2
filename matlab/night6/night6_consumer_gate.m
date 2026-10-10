function [status, reason] = night6_consumer_gate(consumer, held, elsewhere, subset)
% NIGHT6_CONSUMER_GATE  Whether one consumer runs from this mask file - the ONE site.
%
%   [status, reason] = night6_consumer_gate(consumer, held, elsewhere, subset)
%
%   consumer   a consumer name ('spikes', 'hrv', ...)
%   held       the decoded heldconsumers_json (RULING 2026-10-10 item 1: the hold is per
%              consumer): a field per consumer QC holds in this recording, with its reasons
%              and, when Andrea (or the pilot's stand-in run) released it, 'released'
%   elsewhere  the decoded elsewhere_json (RULING 2026-10-10 item 5): a field per consumer
%              this file does not finish, naming the file that does
%   subset     the batch list's consumer subset for this recording; {} runs every consumer
%
% status is '' (no objection - the plan's own checks decide) or, in this order:
%   'skipped_elsewhere'     the consumer is finished in another mask file (its masks are
%                           not in this one)
%   'skipped_not_in_batch'  the batch runs a subset of the consumers for this recording
%   'skipped_held'          QC holds the consumer in this recording and it is not released;
%                           the other consumers run (never the whole recording)
    status = '';
    reason = '';
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
    if isfield(held, consumer)
        h = held.(consumer);
        released = isstruct(h) && isfield(h, 'released') && ~isempty(strtrim(char(h.released)));
        if ~released
            r = {};
            if isstruct(h) && isfield(h, 'reasons'), r = cellstr(h.reasons); end
            status = 'skipped_held';
            reason = ['QC holds this consumer in this recording (RULING 2026-10-10 item 1): ' ...
                      strjoin(r, '; ')];
        end
    end
end
