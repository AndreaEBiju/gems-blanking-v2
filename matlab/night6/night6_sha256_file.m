function h = night6_sha256_file(f)
% NIGHT6_SHA256_FILE  SHA-256 of a file's raw bytes, lower-case hex. Declared once.
%
%   h = night6_sha256_file(f)
%
% The one hash the wrapper records (mask files, called functions, the step1a fallback
% list). It hashes the bytes on disk, never decoded text: a CRLF copy, a BOM or a
% different encoding of the same characters is a different file and hashes differently,
% exactly as Python's hashlib.sha256(path.read_bytes()) does.
    fid = fopen(f, 'r');
    if fid < 0
        error('night6:hashOpen', 'cannot open %s to hash it', f);
    end
    closer = onCleanup(@() fclose(fid));
    b = fread(fid, inf, '*uint8');
    md = java.security.MessageDigest.getInstance('SHA-256');
    md.update(typecast(b, 'int8'));
    h = lower(reshape(dec2hex(typecast(md.digest(), 'uint8'), 2)', 1, []));
end
