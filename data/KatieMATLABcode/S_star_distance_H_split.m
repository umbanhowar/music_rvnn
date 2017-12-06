function [partial_dist_LcR] = ...
    S_star_distance_H_split(L_num_blocks, partial_distmat_LcR)

% S_STAR_DISTANCE_H_SPLIT uses L_NUM_BLOCKS and PARTIAL_DISTMAT_LCR to
% compute the distance between two matrices L and R used to create
% PARTIAL_DISTMAT_LCR and L_NUM_BLOCKS. This code runs if the number of
% rows in R exceeds 5. In S_STAR_DISTANCE_H_SPLIT, we split up the distance
% computation so that we do not hit the memory wall. For this function, we
% assume that L has to have at least the same number of rows as R. This
% computation involves recursion.
%
% INPUT: L_NUM_BLOCKS -- Number of repeats in each row of matrix L
%        PARTIAL_DISTMAT_LCR -- Pairwise distances between rows of L and
%                               rows of R
%
% OUTPUT: PARTIAL_DIST_LCR -- Distance between L and R

% Note: S_STAR_DISTANCE_REPS_SPLIT is very similar to
%       S_STAR_DISTANCE_H_SPLIT

% Step 0: Initialize the temporary variables
num_L_rows = size(partial_distmat_LcR,1);
num_R_rows = size(partial_distmat_LcR,2);

% Split rows in R into two pieces: one piece with at most five partial
%       representations and a second piece. 
num_bins = ceil(num_R_rows/5);
split_R = round(num_R_rows/num_bins); 

inds_vec = [1:num_L_rows];
min_val = [];

% Note: We are going to split the computation into two parts and add the 
%       results of those two computations together. We will denote these
%       two computations with variables with names  *_FIRST and *_SECOND. 

% Split rows in R into two lists.
R_first_inds = [1:split_R];
R_second_inds = [(split_R+1):num_R_rows];

% Split rows in L into two lists based on indices assigned in each loop 
%       iteration. Before beginning the loop, we need to determine which
%       rows in L will be compared to rows in R. We encode this information
%       in the variable L_FIRST_LST.

L_first_lst = nchoosek(inds_vec,split_R);
num_loops = size(L_first_lst,1);

% At the end of each iteration, we will do the computations on the two
%       parts of the two lists and add the results together. Then we will
%       add this resulting sum to LOOP_VEC.
for i = 1:num_loops
    % Split the indices into two lists
    L_first_inds = L_first_lst(i,:);
    L_second_inds = setdiff(inds_vec,L_first_inds);
    
    % Using the two lists of indices, split L and the associated 
    %       information into two lists
    L_num_blocks_first = L_num_blocks(L_first_inds);
    L_num_blocks_second = L_num_blocks(L_second_inds);
    
    % Using the indices for splitting the partial representations, isolate
    %       the relevant entries in DISTMAT_LCR
    dist_first = partial_distmat_LcR;
    dist_first(L_second_inds,:) = [];
    dist_first(:,R_second_inds) = [];
    
    dist_second = partial_distmat_LcR;
    dist_second(L_first_inds,:) = [];
    dist_second(:,R_first_inds) = [];
    
    % Complete the two computations
    comp_one = S_star_partial_dist(L_num_blocks_first, dist_first);
    if (num_R_rows - split_R) > 5
        % If the second half of the split contains more than 5
        %       representations, RECURSE
        comp_two = ...
            S_star_distance_H_split(L_num_blocks_second, dist_second);
    else
        comp_two = S_star_partial_H_dist(L_num_blocks_second, dist_second);
    end
    
    % Add the two computations together and determine if the new value is
    %       smaller than the previous minimum value
    combine_dist = comp_one + comp_two;
    min_val = min([min_val, combine_dist]);
end

partial_dist_LcR = min_val;

end