function opts = night6_mmc_opts(beatsFile, fs, n)
% NIGHT6_MMC_OPTS  extract_mmc options with the R-peak UNIT DECLARED, and checked.
%
%   opts = night6_mmc_opts(beatsFile, fs, n)
%
% beatsFile  the epoch's beats file (heartlocs: 1-based SAMPLE indices into the epoch
%            at fs; written by night6_run_recording)
% fs, n      the gastric input's sample rate and length
%
% RULING 2026-10-08 (g) 2 / invariant 14: units are declared, never left to a default.
% extract_mmc reads opts.rpeakVar from opts.rpeakFile and treats it as SECONDS unless
% opts.rpeakUnits starts with 'sample' - tolerance_sweep.m passed sample-index
% heartlocs with no unit, so every R-peak landed past the end of the signal and
% cardiac blanking was off. Here the variable, its file, its unit and its sample rate
% are named together, and extract_mmc's own conversion (extract_mmc.m, "R-peak
% times" block: rT = pk / rpeakFs for samples, rT = pk otherwise; then
% rIdx = round(rT * fs)) is reproduced and REQUIRED to land every R-peak back on its
% own sample inside [1, n]. A unit that disagrees with the data moves each R-peak by
% a factor of fs and fails here, before the call. opts.rpeakTimes is never set: it
% would take precedence over rpeakVar and bypass the declared unit.
    B = load(beatsFile, 'heartlocs', 'fs');
    if abs(double(B.fs) - fs) > 1e-9
        error('night6:rpeakFs', 'beats at fs %.6f, gastric input at fs %.6f', double(B.fs), fs);
    end
    opts = struct('gastricCols', 1:3, 'rpeakVar', 'heartlocs', 'rpeakFile', beatsFile, ...
                  'rpeakUnits', 'samples', 'rpeakFs', fs);
    pk = double(B.heartlocs(:));
    if startsWith(lower(opts.rpeakUnits), 'sample')
        rT = pk / opts.rpeakFs;
    else
        rT = pk;
    end
    r = round(rT * fs);
    if ~isequal(r, pk) || any(r < 1 | r > n)
        error('night6:rpeakUnits', ['extract_mmc would place the R-peaks at samples ' ...
              '%s..., not at heartlocs %s...: rpeakUnits ''%s'' does not match the data'], ...
              mat2str(r(1:min(3, end))'), mat2str(pk(1:min(3, end))'), opts.rpeakUnits);
    end
end
