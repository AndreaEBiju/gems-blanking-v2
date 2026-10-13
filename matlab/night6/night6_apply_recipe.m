function x = night6_apply_recipe(r, Y, rows)
% NIGHT6_APPLY_RECIPE  One signal's samples from the file's columns, in the file's units.
%
%   x = night6_apply_recipe(r, Y, rows)
%
%   r     a recipe from night6_signal_recipe
%   Y     the recording, nFile x nChannels, in the declared units
%   rows  the file rows to form (1-based)
%
% Returns numel(rows) x 1 double. Raw, pairs-lead and tripole recipes are the weighted sum
% Y(rows, cols) * weights. A 'stomach_ref' recipe is applied in the order of operations of
% the derivation that made the masks' stomach_ref (gems_blanking_v2.derive.derivations.
% build_stomach_reference, specified by derive.stomach_recipe.apply_stomach_ref_recipe):
% the positive contact minus the NaN-omitting mean of the contacts, the sum taken left to
% right ((c1 + c2) + c3) over the finite contacts and divided by their count - which is
% what numpy.nanmean computes - so the result equals Python's sample for sample. Its
% recorded weights are checked by night6_signal_recipe but NOT applied as a product: a dot
% product rounds differently in the last bit.
    C = double(Y(rows, r.cols));
    if ~strcmp(r.kind, 'stomach_ref')
        x = C * r.weights;
        return
    end
    switch r.form
        case 'passthrough'
            x = C(:, r.pos);
        case 'positive_minus_nanmean'
            finite = ~isnan(C);
            Z = C;
            Z(~finite) = 0;
            s = Z(:, 1);
            for k = 2:size(Z, 2)
                s = s + Z(:, k);
            end
            x = C(:, r.pos) - s ./ sum(finite, 2);
        otherwise
            error('night6:stomachRefForm', 'stomach_ref form %s is not one Night 6 applies', r.form);
    end
end
