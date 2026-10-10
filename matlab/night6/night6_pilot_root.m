function root = night6_pilot_root(pilotRoot, outRoot)
% NIGHT6_PILOT_ROOT  The declared pilot folder, checked; '' when no pilot run is declared.
%
%   root = night6_pilot_root('', outRoot)          -> ''  (not a pilot run)
%   root = night6_pilot_root(pilotRoot, outRoot)   -> the pilot folder's real path
%
% Only a DECLARED pilot run (night6_batch list "pilot_run": true with "pilot_root"; the
% night6_run_recording name-value PilotRoot) may read masks made on the STAND-IN tolerance
% table (RULING 2026-10-09 (g) 7: "written to the pilot folder only") or take its beat
% trains from a beats_root (review 2026-10-10 fixes 1-3). The MATLAB twin of
% gems_blanking_v2.extent.tolerance.check_pilot_root / check_stand_in_outputs. Refused by
% name ('night6:pilotRoot') when:
%   - pilotRoot is not an existing folder;
%   - it is, or sits under, a reparse point - a junction or a symbolic link - judged on
%     every component of the path AS GIVEN (made absolute, never resolved, since resolving
%     follows the link): such a folder can name another folder (production) while its path
%     still reads "pilot";
%   - outRoot does not lie inside it, both compared by their REAL paths (Java NIO
%     toRealPath: 8.3 aliases, case and links resolved; an outRoot not yet created is
%     resolved through its nearest existing ancestor).
    root = '';
    pilotRoot = char(pilotRoot);
    if isempty(pilotRoot), return, end
    if ~isfolder(pilotRoot)
        error('night6:pilotRoot', 'the declared pilot_root %s is not an existing folder', ...
              pilotRoot);
    end
    q = java.io.File(pilotRoot).getAbsoluteFile().toPath().normalize();
    while ~isempty(q)
        if is_reparse(q)
            error('night6:pilotRoot', ['the declared pilot_root %s is or sits under a reparse ' ...
                  'point (%s: a junction or symbolic link), which can name another folder; ' ...
                  'refused'], pilotRoot, char(q.toString()));
        end
        q = q.getParent();
    end
    root = real_path(pilotRoot);
    out = real_path(outRoot);
    if ~is_under(out, root)
        error('night6:pilotRoot', ['out_root %s (%s) is not inside the declared pilot folder ' ...
              '%s (%s): a pilot run writes under its pilot folder only'], char(outRoot), out, ...
              pilotRoot, root);
    end
end

function tf = is_reparse(p)
% The path itself (not its target): a symbolic link, or a reparse point Java reports as
% "other" (a junction - WindowsFileAttributes.isOther is true for a non-symlink reparse point).
    opts = javaArray('java.nio.file.LinkOption', 1);
    opts(1) = java.nio.file.LinkOption.NOFOLLOW_LINKS;
    cls = java.lang.Class.forName('java.nio.file.attribute.BasicFileAttributes');
    a = java.nio.file.Files.readAttributes(p, cls, opts);
    tf = a.isSymbolicLink() || a.isOther();
end

function r = real_path(p)
    P = java.io.File(char(p)).getAbsoluteFile().toPath().normalize();
    none = javaArray('java.nio.file.LinkOption', 0);
    rest = {};
    while ~java.nio.file.Files.exists(P, none)
        leaf = P.getFileName();
        P = P.getParent();
        if isempty(P) || isempty(leaf)
            error('night6:pilotRoot', 'no part of %s exists', char(p));
        end
        rest = [{char(leaf.toString())}, rest]; %#ok<AGROW>
    end
    r = char(P.toRealPath(none).toString());
    if ~isempty(rest), r = fullfile(r, rest{:}); end
end

function tf = is_under(p, root)
    strip = @(s) regexprep(s, '[\\/]+$', '');
    p = strip(p); root = strip(root);
    if ispc, p = lower(p); root = lower(root); end
    tf = strcmp(p, root) || startsWith(p, [root filesep]);
end
