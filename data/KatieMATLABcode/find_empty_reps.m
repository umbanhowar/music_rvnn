function [num_empty_reps, empty_rep_inds] = ...
    find_empty_reps(num_shingles, data_type)

% FIND_EMPTY_REPS searches a directory, returning the number of files
% without any partial representations. 
%
% INPUT: NUM_SHINGLES -- Number of feature vectors per shingle
%        DATA_TYPE -- Flag for the kind of data we are checking. 
%                      'A' means audio and 'S' (default) means score
%
% OUTPUT: NUM_EMPTY_REPS -- Number of files without any partial
%                           representations
%         EMPTY_REP_INDS -- Index number of files without partial
%                           representations

if nargin < 2
    data_type = 'S';
end

key_dir = '/Users/katiek/Desktop/CurrentCode/FileKeys/';
if strcmp(data_type,'S')
    compare_songlist = importdata([key_dir, ...
        'filenameMAT_list_ShingleNumber', num2str(num_shingles),'.csv']);
elseif strcmp(data_type,'A')
    compare_songlist = importdata([key_dir, ...
        'filenameAUDIO_list_ShingleNumber', num2str(num_shingles),'.csv']);
end    

num_songs = length(compare_songlist);
num_empty_reps = 0;
empty_rep_inds = [];

for i = 1:num_songs
    % Load i-th file and determine the number of partial representations
    file_i = compare_songlist{i};
    songI = load(file_i);
    
    I_num_reps = songI.num_partials;
    
    if I_num_reps == 0
        num_empty_reps = num_empty_reps + 1;
        empty_rep_inds = [empty_rep_inds;i];
    end
    
    clear file_i
    clear SongI
    clear I_num_reps
end