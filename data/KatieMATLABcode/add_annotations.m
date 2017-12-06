function [anno_lst] = add_annotations(input_mat, song_length)

% ADD_ANNOTATIONS adds annotation markers to the pairs of repeats in
% INPUT_MAT.
%
% INPUT: INPUT_MAT -- List of pairs of repeats possibly with annotations
%                     marked. The first two columns refer to the first
%                     repeat of the pair, the second two refer to the
%                     second repeat of the pair, the fifth column refers
%                     to the length of the repeats, and the sixth column
%                     contains any annotations, which will be immediately
%                     removed. 
%        SONG_LENGTH -- The number of audio shingles in the song
%
% OUTPUT: ANNO_LST -- List of pairs of repeats with annotations marked

% Initialize the inputs
M = input_mat;
n = size(M,1);
sn = song_length;
% Scrub out all annotation markers that may or may not already exist
M(:,6) = 0;

% Find where all repeats start
SI = M(:,1);
SJ = M(:,3);

% Generate a matrix with 1's at each pair (SI,SJ) and 0's elsewhere
SS = ones(n,1); % Needed for the SPARSE() command
S = sparse(SI,SJ,SS,sn,sn);
full_S = full(S); % FULL_S is an upper triangular matrix. We need a 
                  % symmetric matrix for STITCH_DIAGS, which we create in
                  % the next line.
full_S = full_S + full_S'; 

% Stitch the information from INPUT_MAT into one row (where each entry
% represents a time step and the given number denotes which group
% that time step is a member of)
song_pattern = stitch_diags(full_S,0);
SPmax = max(song_pattern);

% Now add the annotation markers to the list of pairs of repeats
for s = 1:SPmax % Loop over all group annotations
    % Find all repeats with the annotation S and mark the time steps where
    % these repeats begin with S
    pinds = find(song_pattern == s);
    check_inds = (M(:,6) == 0); % 1 if annotation is not yet marked
                                % 0 if annotation is already marked
                                
    % Loop over all found time steps now marked with annotation S
    for j = pinds 
        % Find all starting pairs that contain time step J AND do NOT
        % have an annotation
        mark_inds = (SI == j) + (SJ == j);
        mark_inds = (mark_inds > 0);
        mark_inds = check_inds.*mark_inds;
        % Add the found annotations to the pairs of repeats that contain
        % the time step J
        M(:,6) = (M(:,6) + s*mark_inds);
        % Remove the pairs of repeats that you have just added annotation
        % markers to from consideration
        check_inds = check_inds - mark_inds;
    end
end

[~,temp_inds] = sort(M(:,6), 'ascend');
anno_lst = M(temp_inds,:);

end