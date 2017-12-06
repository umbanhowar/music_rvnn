function process_many_dirs(shingleStart, shingleEnd, thresh_maxval, ...
    thresh_flag, print_flag, clock_flag)

% PROCESS_MANY_DIRS processes directories of files for a range of shingle
% lengths (that is the number of feature vectors per shingle)
%
% INPUT: SHINGLESTART -- Smallest number of feature vectors per shingle
%        SHINGLESEND -- Largest number of feature vectors per shingle
%        THRESH_MAXVAL -- Maximum threshold value
%        THRESH_FLAG -- Denotes which kind of value we are basing our
%                       threshold on: 'p' for proportion of the length of
%                       the orthogonal projection compared to the norm of
%                       an audio shingle, 'a' for angle between the audio
%                       shingles, and 't' for a threshold set without
%                       direct consideration of either 'a' or 'p'
%        PRINT_FLAG -- Flag that determines if the file number that is
%                      being processed gets printed. 0 (default) means the
%                      file numbers will be not printed and 1 means that
%                      the file numbers will be printed
%        CLOCK_FLAG -- Flag that determines if the time it takes to process
%                      one directory will be printed. 1 (default) means the
%                      time will be printed and 0 means that the time will
%                      not be printed
%
% OUTPUT: None. Files are saved.

if nargin < 5
    print_flag = 0;
end

if nargin < 6
    clock_flag = 1;
end

% Compute the threshold given THRESH_MAXVAL and THRESH_FLAG
[thresh] = find_thresh(thresh_flag, thresh_maxval);

% Make directories for each shingle size
make_shingle_dir(shingleStart,shingleEnd)

for i = [shingleStart:shingleEnd]
    num_shingles = i;
    fprintf(['Processing Directory/ies with Shingle Numbers = ', ...
        num2str(num_shingles), '\n']);
    tic;
    process_dir2partials_save(num_shingles, print_flag, thresh)
    if clock_flag == 1
        toc;
    end
end