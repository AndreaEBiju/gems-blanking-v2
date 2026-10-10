function rc = night6_common_start_rec(rec)
% NIGHT6_COMMON_START_REC  The recovery record of the COMMON-START version - the ONE site.
%
%   rc = night6_common_start_rec(rec)
%
% RULING 2026-10-10 item 7 (dose-response safeguard): for comparisons between stim levels,
% every stim_recovery file of an animal starts each analysis at the latest start among them,
% measured from each file's mechanical end (Python extent.recovery_start.COMMON_START_RULE).
% The common-start version RE-USES the run's outputs and cuts every output variable at its
% cut's common start instead of its own: rc is rec (night6_recovery_lead_in, applies true)
% with every cut's start_s, start_sample0 and output_rows_before_start replaced by its
% common_start_s, common_start_sample0 and output_rows_before_common, and version
% 'common_start'. The common start is never before the own start (night6_recovery_start
% refuses that), so the common version only drops more of the early recovery.
    if ~isstruct(rec) || ~isfield(rec, 'applies') || ~rec.applies || ~isfield(rec, 'cuts')
        error('night6:commonStart', 'the common-start version needs an applying recovery record');
    end
    rc = rec;
    for k = 1:numel(rc.cuts)
        c = rc.cuts{k};
        if ~all(isfield(c, {'common_start_s', 'common_start_sample0', ...
                            'output_rows_before_common'}))
            error('night6:commonStart', 'cut %s has no common start (starts v6)', char(c.cut));
        end
        c.own_start_sample0 = c.start_sample0;
        c.start_s = c.common_start_s;
        c.start_sample0 = c.common_start_sample0;
        c.output_rows_before_start = c.output_rows_before_common;
        rc.cuts{k} = c;
    end
    rc.version = 'common_start';
end
