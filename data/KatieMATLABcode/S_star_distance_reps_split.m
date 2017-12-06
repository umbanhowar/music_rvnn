function [dist_AcB_reps] = ...
    S_star_distance_reps_split(A_partial_num_blocks, distmat_AcB)

% S_STAR_DISTANCE_REPS_SPLIT uses A_NUM_BLOCKS and PARTIAL_DISTMAT_ACB to
% compute the distance two lists of partial representations A_PARTIAL_REPS
% and B_PARTIAL_REPS used to create PARTIAL_DISTMAT_ACB and A_NUM_BLOCKS.
% This code runs if the number of partial representations in B_PARTIAL_REPS
% exceeds 5. In S_STAR_DISTANCE_REPS_SPLIT, we split up the distance
% computation so that we do not hit the memory wall. We assume that
% A_PARTIAL_REPS has at least the same number of partial representations as
% B_PARTIAL_REPS. This computation involves recursion.
%
% INPUT: A_PARTIAL_NUM_BLOCKS -- Number of repeats in each partial
%                                representation in A_PARTIAL_REPS
%        DISTMAT_ACB -- Pairwise distances between the partial
%                       representations in A_PARTIAL_REPS and those in
%                       B_PARTIAL_REPS
%
% OUTPUT: DIST_ACB_REPS -- Distance between two lists of partial
%                          representations A_PARTIAL_REPS and
%                          B_PARTIAL_REPS

% Note: S_STAR_DISTANCE_REPS_SPLIT is very similar to
%       S_STAR_DISTANCE_H_SPLIT


% Step 0: Initialize the temporary variables
num_A_reps = size(distmat_AcB,1);
num_B_reps = size(distmat_AcB,2);

% Split the B_PARTIAL_REPS into two pieces: one piece with at most five
%       partial representations and a second piece. 
num_bins = ceil(num_B_reps/5);
split_B = round(num_B_reps/num_bins); 

inds_vec = [1:num_A_reps];
min_val = [];

% Note: We are going to split the computation into two parts and add the 
%       results of those two computations together. We will denote these
%       two computations with variables with names  *_FIRST and *_SECOND. 

% Split B_PARTIAL_REPS into two lists
B_first_inds = [1:split_B];
B_second_inds = [(split_B+1):num_B_reps];

% Split A_PARTIAL_REPS into two lists based on indices assigned in each
%       loop iteration. Before beginning the loop, we need to determine
%       which partial representations in A_PARTIAL_REPS will be compared to
%       B_PARTIAL_REPS. We encode this information in the variable
%       A_FIRST_LST.

A_first_lst = nchoosek(inds_vec,split_B);
num_loops = size(A_first_lst,1);

% At the end of each iteration, we will do the computations on the two
%       parts of the two lists and add the results together. Then we will
%       add this resulting sum to LOOP_VEC.
for i = 1:num_loops
    % Split the indices into two lists
    A_first_inds = A_first_lst(i,:);
    A_second_inds = setdiff(inds_vec,A_first_inds);
    
    % Using the two lists of indices, split A_PARTIAL_REPS and the
    %       associated infromation into two lists. 
    A_num_blocks_first = A_partial_num_blocks(A_first_inds);
    A_num_blocks_second = A_partial_num_blocks(A_second_inds);
    
    % Using the indices for splitting the partial representations, isolate 
    %       the relevant entries in DISTMAT_ACB
    dist_first = distmat_AcB;
    dist_first(A_second_inds,:) = [];
    dist_first(:,B_second_inds) = [];
    
    dist_second = distmat_AcB;
    dist_second(A_first_inds,:) = [];
    dist_second(:,B_first_inds) = [];
    
    % Complete the two computations 
    comp_one = S_star_partial_dist(A_num_blocks_first, dist_first);
    if (num_B_reps - split_B) > 5
        % If the second half of the split contains more than 5
        %       representations, RECURSE
        comp_two = ...
            S_star_distance_reps_split(A_num_blocks_second, dist_second);
    else
        comp_two = S_star_partial_dist(A_num_blocks_second, dist_second);
    end
    
    % Add the two computations together and determine if the new value is
    %       smaller than the previous minimum value
    combine_dist = comp_one + comp_two;
    min_val = min([min_val, combine_dist]);
end

dist_AcB_reps = min_val;

end