function check_recipe_hash(caseFile, outFile)
% CHECK_RECIPE_HASH  Harness for tests/test_night6_recipe_hash.py (gems-blanking-v2).
%
% night6_recipe_sha256 on provenance texts written by Python: per case {text, applied} ->
% the hash it returns, or the error id it raises. Writes JSON.
    C = jsondecode(fileread(caseFile));
    addpath(C.night6);
    out = struct();
    for f = fieldnames(C.cases)'
        c = C.cases.(f{1});
        r = struct('hash', '', 'error_id', '');
        try
            r.hash = night6_recipe_sha256(c.text, logical(c.applied));
        catch ME
            r.error_id = ME.identifier;
        end
        out.(f{1}) = r;
    end
    fid = fopen(outFile, 'w', 'n', 'UTF-8');
    fwrite(fid, jsonencode(out), 'char');
    fclose(fid);
end
