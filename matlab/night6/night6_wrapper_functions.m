function names = night6_wrapper_functions(here)
%NIGHT6_WRAPPER_FUNCTIONS  The wrapper's own functions: every .m file in HERE (this folder)
%   that night6_run_recording reaches, directly or through another of them.
%
%   Derived, never listed by hand (invariant 32: a hand list missed seven helpers, found
%   2026-10-12 in review): starting from night6_run_recording.m, each file of this folder
%   whose name appears as a whole word in a reached file's text is reached. A name met only
%   in a comment is included too - that costs one more hash and shadow check, never a miss.
%   function_provenance refuses any of them resolving outside HERE and records each hash.
%
%   NAMES is a sorted cellstr, without night6_run_recording itself.
    files = dir(fullfile(here, '*.m'));
    stems = sort(erase({files.name}, '.m'));
    self = 'night6_run_recording';
    names = {};
    todo = {self};
    while ~isempty(todo)
        txt = fileread(fullfile(here, [todo{1} '.m']));
        todo(1) = [];
        for s = stems
            if strcmp(s{1}, self) || ismember(s{1}, names)
                continue
            end
            if ~isempty(regexp(txt, ['\<' s{1} '\>'], 'once'))
                names{end + 1} = s{1}; %#ok<AGROW>
                todo{end + 1} = s{1}; %#ok<AGROW>
            end
        end
    end
    names = sort(names);
end
