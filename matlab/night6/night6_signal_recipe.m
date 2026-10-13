function r = night6_signal_recipe(sig, labels, channels, srRecipe)
% NIGHT6_SIGNAL_RECIPE  How one consumer signal is formed from the file's columns.
%
%   r = night6_signal_recipe(sig, labels, channels, srRecipe)
%
%   sig       the signal name ('ANT1', 'L_T', 'LVN2-RVN2', 'stomach_ref', ...)
%   labels    the recording file's chanlabels, in column order (cellstr)
%   channels  meta.json "channels" as a cell of structs (geometry: cuff_id, contact_index,
%             role, config)
%   srRecipe  the mask provenance's stomach_ref_recipe (jsondecode), or [] when the
%             provenance carries none
%
% r is applied by night6_apply_recipe (in the file's units; the caller scales to volts):
%   'raw'      one column                          r.cols, r.weights = 1
%   'pair'     a pairs lead plus - minus           r.cols, r.weights = [1; -1]
%   'tripole'  software tripole, task 04           T = 0.5 V1 - V2 + 0.5 V3 (naive by ruling)
%   'stomach_ref'  the recipe the MASK PROVENANCE records (Andrea, 2026-10-12), applied as
%              recorded and never re-derived here: form 'positive_minus_nanmean' (the
%              positive contact minus the NaN-omitting mean of the contacts, in the
%              derivation's order of operations) or 'passthrough' (one contact). Refused by
%              name (night6:stomachRef*) when the provenance carries no recipe, when a field
%              is missing, when the schema, form or notch is one this reader does not
%              reproduce (a notched stomach_ref is refused: no recording Night 6 runs needs
%              one yet), when its weights are not its form's, when it has 8 or more
%              contacts (numpy's mean sums those pairwise, so the order would not be the
%              derivation's: night6:stomachRefTooMany), or when its contacts are not the
%              file's stomach channels in meta.json order under its declared config.
%              stomach_ref is never formed from a nerve channel, and its masks are never
%              moved onto one.
    col = @(lab) find_one(labels, lab, sig);
    tok = regexp(sig, '^([A-Za-z]+)_T$', 'tokens', 'once');
    if strcmp(sig, 'stomach_ref')
        r = stomach_ref(srRecipe, col, channels);
    elseif contains(sig, '-')
        parts = strsplit(sig, '-');
        r = struct('kind', 'pair', 'cols', [col(parts{1}) col(parts{2})], 'weights', [1; -1]);
    elseif ~isempty(tok)
        % Software tripole, task 04: T = a*V1 + b*V3 - V2 with a = b = 0.5 (applied
        % weights are naive by ruling; never fitted). Contacts from meta.json.
        cuff = tok{1};
        cols = zeros(1, 3);
        for k = 1:3
            hit = cellfun(@(c) isfield(c, 'cuff_id') && strcmp(c.cuff_id, cuff) && ...
                isfield(c, 'contact_index') && c.contact_index == k && ...
                strcmp(c.config, 'independent'), channels);
            if nnz(hit) ~= 1
                error('night6:tripole', '%s: cuff %s needs exactly one independent contact %d', ...
                      sig, cuff, k);
            end
            cols(k) = col(char(channels{hit}.label));
        end
        r = struct('kind', 'tripole', 'cols', cols, 'weights', [0.5; -1; 0.5]);
    else
        r = struct('kind', 'raw', 'cols', col(sig), 'weights', 1);
    end
end

function r = stomach_ref(R, col, channels)
    if isempty(R)
        error('night6:stomachRefRecipe', ['signal stomach_ref: the mask provenance carries no ' ...
              'stomach_ref_recipe, so how the masks'' stomach_ref was built is unknown. Refused ' ...
              '(Andrea, 2026-10-12: the recipe comes from the mask provenance); re-emit the masks']);
    end
    need = {'schema', 'how', 'config', 'contacts', 'positive', 'form', 'weights', 'notch'};
    lacking = need(~isfield(R, need));
    if ~isempty(lacking)
        error('night6:stomachRefRecipe', 'stomach_ref_recipe lacks %s', strjoin(lacking, ', '));
    end
    if ~strcmp(char(R.schema), 'stomach_ref_recipe/1')
        error('night6:stomachRefRecipe', 'stomach_ref_recipe schema %s is not stomach_ref_recipe/1', ...
              char(R.schema));
    end
    if ~(ischar(R.notch) && strcmp(R.notch, 'none'))
        error('night6:stomachRefNotch', ['stomach_ref_recipe names a notch: a notched ' ...
              'stomach_ref is not reproduced by Night 6, so it is refused rather than built ' ...
              'without its notch']);
    end
    form = char(R.form);
    how = char(R.how);
    if ~((strcmp(form, 'positive_minus_nanmean') && strcmp(how, 'common_average')) || ...
         (strcmp(form, 'passthrough') && ismember(how, {'hardware', 'single_channel'})))
        error('night6:stomachRefForm', 'stomach_ref_recipe how %s / form %s is not one Night 6 applies', ...
              how, form);
    end
    contacts = cellstr(R.contacts);
    contacts = contacts(:)';
    n = numel(contacts);
    pos = find(strcmp(contacts, char(R.positive)));
    if n < 1 || numel(unique(contacts)) ~= n || ~isequal(pos, 1)
        error('night6:stomachRefContacts', ['stomach_ref_recipe contacts [%s] must be distinct ' ...
              'and start with the positive input %s'], strjoin(contacts, ' '), char(R.positive));
    end
    if n >= 8
        % numpy.nanmean over 8 or more columns sums them pairwise, not left to right, so
        % night6_apply_recipe's order would no longer be the derivation's (measured: equal
        % for 2..7 contacts, unequal from 8). Refused rather than built approximately.
        error('night6:stomachRefTooMany', ['stomach_ref_recipe has %d contacts: from 8 on, ' ...
              'numpy sums the mean pairwise, so Night 6 cannot reproduce it sample for ' ...
              'sample'], n);
    end
    if strcmp(form, 'passthrough') && n ~= 1
        error('night6:stomachRefContacts', 'a passthrough stomach_ref reads one contact, not [%s]', ...
              strjoin(contacts, ' '));
    end
    isStomach = cellfun(@(c) isfield(c, 'role') && strcmp(c.role, 'stomach'), channels);
    metaStomach = cellfun(@(c) char(c.label), channels(isStomach), 'UniformOutput', false);
    if strcmp(form, 'positive_minus_nanmean') && ~isequal(contacts, metaStomach(:)')
        error('night6:stomachRefContacts', ['stomach_ref_recipe averages [%s] but the file''s ' ...
              'stomach channels (meta.json order) are [%s]'], strjoin(contacts, ' '), ...
              strjoin(metaStomach, ' '));
    end
    if strcmp(form, 'passthrough') && (isempty(metaStomach) || ~strcmp(contacts{1}, metaStomach{1}))
        error('night6:stomachRefContacts', ['stomach_ref_recipe passes %s through but the file''s ' ...
              'stomach channels are [%s]'], contacts{1}, strjoin(metaStomach, ' '));
    end
    cfg = cellfun(@(c) char(c.config), channels(isStomach), 'UniformOutput', false);
    if ~all(strcmp(cfg, char(R.config)))
        error('night6:stomachRefConfig', ['stomach_ref_recipe config %s, but meta.json declares ' ...
              '[%s] for the stomach channels'], char(R.config), strjoin(cfg, ' '));
    end
    w = double(R.weights(:));
    if strcmp(form, 'positive_minus_nanmean')
        want = double((1:n)' == pos) - 1 / n;      % the derivation's impulse responses
    else
        want = 1;
    end
    if ~isequal(w, want)
        error('night6:stomachRefWeights', ['stomach_ref_recipe weights [%s] are not the %s ' ...
              'weights [%s]'], num2str(w', 17), form, num2str(want', 17));
    end
    cols = zeros(1, n);
    for k = 1:n
        cols(k) = col(contacts{k});
    end
    r = struct('kind', 'stomach_ref', 'cols', cols, 'weights', w, 'form', form, 'pos', pos);
end

function i = find_one(labels, lab, sig)
    i = find(strcmp(labels, lab));
    if numel(i) ~= 1
        error('night6:label', 'signal %s needs channel %s, found %d in the file', sig, lab, numel(i));
    end
end
