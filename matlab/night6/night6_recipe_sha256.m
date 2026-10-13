function h = night6_recipe_sha256(provJson, applied)
% NIGHT6_RECIPE_SHA256  SHA-256 of the stomach_ref_recipe a Night 6 result applied, or 'none'.
%
%   h = night6_recipe_sha256(provJson, applied)
%
%   provJson  the mask file's provenance_json text, exactly as stored (char)
%   applied   true when the run formed stomach_ref (plan.recipes.stomach_ref exists)
%
% Andrea, 2026-10-12: every Night 6 result records the sha256 of the stomach_ref_recipe it
% applied, or "none". ONE canonical form on both sides of the MATLAB/Python boundary
% (invariant 22): the UTF-8 bytes of the recipe as canonical JSON - sorted keys, ASCII
% escapes, separators "," and ":" - which is exactly how emit.provenance writes it inside
% provenance_json (MaskProvenance.to_json). So MATLAB never re-serialises (its jsonencode
% writes 5.0 as 5): it hashes the bytes of the top-level "stomach_ref_recipe" value as they
% stand in the text, and Python's derive.stomach_recipe.stomach_ref_recipe_sha256 hashes the
% same canonical bytes (round trip tested in tests/test_night6_recipe_hash.py).
%
% Refused by name ('night6:recipeHash'): applied but the text has no top-level recipe, or the
% extracted value does not decode to the recipe jsondecode reads from the whole text.
    if ~applied
        h = 'none';
        return
    end
    t = char(provJson);
    [a, b] = top_level_value(t, 'stomach_ref_recipe');
    if isempty(a)
        error('night6:recipeHash', ['stomach_ref was formed but the provenance has no ' ...
              'top-level stomach_ref_recipe']);
    end
    v = t(a:b);
    P = jsondecode(t);
    if ~isequaln(jsondecode(v), P.stomach_ref_recipe)
        error('night6:recipeHash', 'the extracted stomach_ref_recipe is not the decoded one');
    end
    md = java.security.MessageDigest.getInstance('SHA-256');
    md.update(typecast(unicode2native(v, 'UTF-8'), 'int8'));
    h = lower(reshape(dec2hex(typecast(md.digest(), 'uint8'), 2)', 1, []));
end

function [a, b] = top_level_value(t, key)
% The 1-based span [a, b] of key's value in the top-level object of JSON text t ([] when
% absent): a scan that tracks strings (with escapes) and nesting depth, so a nested key or a
% string that merely contains the key's name is never taken for it.
    a = []; b = [];
    want = ['"' key '"'];
    depth = 0; inStr = false; esc = false; i = 1; n = numel(t);
    while i <= n
        c = t(i);
        if inStr
            if esc, esc = false;
            elseif c == '\', esc = true;
            elseif c == '"', inStr = false;
            end
            i = i + 1;
            continue
        end
        if c == '"'
            if depth == 1 && i + numel(want) - 1 <= n && strcmp(t(i:i + numel(want) - 1), want)
                j = i + numel(want);
                while j <= n && isspace(t(j)), j = j + 1; end
                if j <= n && t(j) == ':'   % a key, not a value
                    j = j + 1;
                    while j <= n && isspace(t(j)), j = j + 1; end
                    a = j;
                    b = value_end(t, j);
                    return
                end
            end
            inStr = true;
        elseif c == '{' || c == '['
            depth = depth + 1;
        elseif c == '}' || c == ']'
            depth = depth - 1;
        end
        i = i + 1;
    end
end

function e = value_end(t, j)
% The last character of the JSON value starting at t(j).
    n = numel(t);
    if t(j) == '{' || t(j) == '['
        d = 0; inStr = false; esc = false;
        for k = j:n
            c = t(k);
            if inStr
                if esc, esc = false;
                elseif c == '\', esc = true;
                elseif c == '"', inStr = false;
                end
            elseif c == '"', inStr = true;
            elseif c == '{' || c == '['
                d = d + 1;
            elseif c == '}' || c == ']'
                d = d - 1;
                if d == 0, e = k; return, end
            end
        end
        error('night6:recipeHash', 'unterminated stomach_ref_recipe value');
    elseif t(j) == '"'
        esc = false;
        for k = j + 1:n
            if esc, esc = false;
            elseif t(k) == '\', esc = true;
            elseif t(k) == '"', e = k; return
            end
        end
        error('night6:recipeHash', 'unterminated stomach_ref_recipe string');
    else
        k = j;
        while k <= n && ~any(t(k) == ',}]'), k = k + 1; end
        e = k - 1;
    end
end
