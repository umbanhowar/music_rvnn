function partial_lst_distmat = dist_partial_lst(compare_songlist)

% DIST_PARTIAL_LIST computes the matrix of pairwise comparisons between the
% representations (or lists of representations) listed in COMPARE_SONGLIST.
% For this comparison, we add the distances between representations of the
% same width.
%
% INPUT: COMPARE_SONGLIST -- List of filenames for the representations (or
%                            lists of representations) that are to be
%                            compared
%
% OUTPUT: PARTIAL_LST_DISTMAT -- Matrix of pairwise comparisons between
%                                the representations (or lists of
%                                representations) listed in
%                                COMPARE_SONGLIST

num_songs = length(compare_songlist);
D = zeros(num_songs);
empty_rep_list_inds = [];

inds2process = [1:(num_songs - 1)];

while ~isempty(inds2process)
    i = inds2process(1);
    % Load the I-th file
    file_i = compare_songlist{i};
    songI = load(file_i);
    
    I_partial_reps = songI.partial_reps;
    I_partial_widths = songI.partial_widths;
    I_partial_keys = songI.partial_keys;
    I_partial_num_blocks = songI.partial_num_blocks;
    I_num_reps = songI.num_partials;
    
    clear file_i
    clear SongI
    
    I_rmlst = [];
    
    for j = (i + 1): num_songs
        % Load the J-th file
        file_j = compare_songlist{j};
        songJ = load(file_j);
        
        J_partial_reps = songJ.partial_reps;
        J_partial_widths = songJ.partial_widths;
        J_partial_keys = songJ.partial_keys;
        J_partial_num_blocks = songJ.partial_num_blocks;
        J_num_reps = songJ.num_partials;
        
        clear file_j
        clear SongJ
        
        % Start the distance computation
        
        % If Song I has no partial representations, then we can not make a 
        %       comparison and D(i,j) is set to NaN (not a number).
        %       Similarly, if Song J has no partial representations, then
        %       D(i,j) is set to NaN (not a number).
        
        if I_num_reps == 0
            D(i,j) = nan;
            empty_rep_list_inds = [empty_rep_list_inds;i];
        elseif J_num_reps == 0
            D(i,j) = nan;
            empty_rep_list_inds = [empty_rep_list_inds;j];
        else
            D(i,j) = dist_two_partial_sets(I_partial_reps, I_partial_keys, ...
                I_partial_widths, I_partial_num_blocks, J_partial_reps, ...
                J_partial_keys, J_partial_widths, J_partial_num_blocks);
        end
        
        % If D(i,J) = 0, then Song I is identical to Song J. Therefore, we
        %       do not to redo the computations that Songs I and J have
        %       already done. We can copy the results from the I-th row of
        %       D into the J-th row of D and we remove the index J from
        %       consideration. Inside this loop, we just add the index J to
        %       the I_RMLST.
        if D(i,j) == 0
            I_rmlst = [I_rmlst;j];
        end
        
        clear J_partial_reps
        clear J_partial_widths
        clear J_partial_keys
        clear J_partial_num_blocks
        clear J_num_reps
    end
    
    % Copy the results from the I-th row of D into the rows of D that 
    %       correspond to the songs that are identical to Song I and remove
    %       and we remove the corresponding indices from consideration.
    if ~isempty(I_rmlst)
        num_rminds = length(I_rmlst);
        for r = 1:num_rminds
            ri = I_rmlst(r);
            D(ri,[(ri+1):num_songs]) = D(i,[(ri+1):num_songs]);
            inds2process(inds2process == ri) = [];
        end
    end
    clear num_rminds; clear ri; clear I_rmlst;
    
    % Remove the current index from INDS2PROCESS and clear the associated
    %       song information. 
    
    inds2process(1) = [];
    
    clear I_partial_reps
    clear I_partial_widths
    clear I_partial_keys
    clear I_partial_num_blocks
    clear I_num_reps
end

max_val = max(max(D));

diag_dist_mat = max_val*eye(num_songs);
for e = 1:length(empty_rep_list_inds)
    diag_dist_mat(empty_rep_list_inds(e),empty_rep_list_inds(e)) = NaN;
end

partial_lst_distmat = D + D' + diag_dist_mat;