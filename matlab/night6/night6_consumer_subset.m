function [subset, tag] = night6_consumer_subset(given, known)
% NIGHT6_CONSUMER_SUBSET  A batch's consumer subset for one recording, checked; its folder tag.
%
%   [subset, tag] = night6_consumer_subset(given, known)
%
% RULING 2026-10-10 item 5: Night 6 runs the non-spike consumers per recording as soon as
% their masks exist and the spike consumer per animal later, so a batch list may name the
% consumers to run per recording. given is a cell/string array of consumer names ({} or ''
% = every consumer); a name not in known is refused by name ('night6:consumers'), and a
% name given twice too. subset is sorted ({} for every consumer). tag is '' for every
% consumer, else '__' and the sorted two-letter codes (breathing br, hrv hv, mmc mm,
% slow_wave sw, spikes sp, velocity ve) - the epoch folder's suffix, so two subsets of one
% mask folder never share an output folder or a record (and the path stays short, rule 5).
    codes = struct('breathing', 'br', 'hrv', 'hv', 'mmc', 'mm', 'slow_wave', 'sw', ...
                   'spikes', 'sp', 'velocity', 've');
    if isempty(given)
        subset = {};
        tag = '';
        return
    end
    given = cellstr(given);
    given = given(:)';
    bad = setdiff(given, known);
    if ~isempty(bad)
        error('night6:consumers', 'unknown consumer(s) [%s]; known: [%s]', ...
              strjoin(bad, ' '), strjoin(known, ' '));
    end
    if numel(unique(given)) ~= numel(given)
        error('night6:consumers', 'a consumer is named twice in [%s]', strjoin(given, ' '));
    end
    subset = sort(given);
    parts = cellfun(@(c) codes.(c), subset, 'UniformOutput', false);
    tag = ['__' strjoin(parts, '_')];
end
