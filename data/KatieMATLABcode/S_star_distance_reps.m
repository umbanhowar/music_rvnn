function [dist_AcB_reps] = S_star_distance_reps(A_partial_reps, ...
    A_partial_keys, A_partial_num_blocks, B_partial_reps, B_partial_keys)

% S_STAR_DISTANCE_REPS computes the distance between two lists of partial
% representations, denoted here as A_PARTIAL_REPS and B_PARTIAL_REPS,
% encoding representations of the same width. Associated to A_PARTIAL_REPS
% are A_PARTIAL_KEYS, and A_PARTIAL_NUM_BLOCKS. Associated to
% B_PARTIAL_REPS is B_PARTIAL_KEYS. We assume that A_PARTIAL_REPS has at
% least the same number of partial representations as B_PARTIAL_REPS.

% INPUT: A_PARTIAL_REPS -- First list of partial hierarchical
%                          representations
%        A_PARTIAL_KEYS -- List of vectors of lengths of repeats in the
%                          partial representations. Each vector of lengths
%                          corresponds to a partial hierarchical
%                          representation stored in A_PARTIAL_REPS
%        A_PARTIAL_NUM_BLOCKS -- Number of repeats in each partial
%                                representation in A_PARTIAL_REPS
%        B_PARTIAL_REPS -- Second list of partial hierarchical 
%                          representations
%        B_PARTIAL_KEYS -- List of vectors of lengths of repeats in the
%                          partial representations. Each vector of lengths
%                          corresponds to a partial hierarchical
%                          representation stored in B_PARTIAL_REPS
% 
% OUTPUT: DIST_ACB_REPS -- Distance between two lists of partial
%                          representations A_PARTIAL_REPS and
%                          B_PARTIAL_REPS

% Note: S_STAR_DISTANCE_REPS is very similar to S_STAR_DISTANCE

% Step 1: Compute the pairwise distance between the representations in
%         A_PARTIAL_REPS and those in B_PARTIAL_REPS
num_A_reps = length(A_partial_reps);
num_B_reps = length(B_partial_reps);

small_dist_AcB = zeros(num_A_reps,num_B_reps); % Preallocate SMALL_DIST_ACB
for a = 1:num_A_reps
    rep_A = A_partial_reps{a};
    key_A = A_partial_keys{a};
    for b = 1:num_B_reps
        rep_B = B_partial_reps{b};
        key_B = B_partial_keys{b};
        small_dist_AcB(a,b) = ...
            dist_between_reps(rep_A, key_A, rep_B, key_B);
    end
end

% Step 2: Find and remove any exact matches from consideration. (We have
%         proven that d_R({A},{B}) = d_R({A^},{B^}) where {A}, {B} are
%         lists of representations and {A^}, {B^} are those lists with the
%         exact matches removed).

[A_exact, B_exact] = find(small_dist_AcB == 0);

A_ne = setdiff([1:num_A_reps],A_exact);
B_ne = setdiff([1:num_B_reps],B_exact);

num_A_reps_ne = length(A_ne);
num_B_reps_ne = length(B_ne);

% Case 1 -- {A} == {B} (The partial representation list A and the partial 
%           representation list B are identical)
if (num_A_reps_ne == 0) && (num_B_reps_ne == 0)
    dist_AcB_reps = 0;
    
% Case 2 -- {A} > {B} (The partial representation list B is contained in 
%           the partial representation list A)
elseif (num_A_reps_ne > 0) && (num_B_reps_ne == 0) 
    A_partial_num_blocks_ne = A_partial_num_blocks(A_ne);
    dist_AcB_reps = sum(A_partial_num_blocks_ne);
    
% Case 3 -- The partial representation list A and the partial 
%           representation list B are not contained in each other
elseif (num_B_reps_ne > 0)
    % Isolate the information for the representations that do not have
    %       exact matches that are needed for the below comparison. 
    A_partial_num_blocks_ne = A_partial_num_blocks(A_ne);
    
    % We also need to isolate the distances in SMALL_DIST_ACB that are
    %       associated with the above isolated representations. First
    %       remove the rows that are associated with partial
    %       representations in A_PARTIAL_REPS that have exact matches in
    %       B_PARTIAL_REPS. Next remove the columns that are associated
    %       with partial representations in B_PARTIAL_REPS that have exact
    %       matches in A_PARTIAL_REPS.
    partial_dist_mat_AcB = small_dist_AcB;
    partial_dist_mat_AcB(A_exact,:) = [];
    partial_dist_mat_AcB(:,B_exact) = [];
    
    % Compute the distance between A_PARTIAL_REPS and B_PARTIAL_REPS
    if (num_B_reps_ne <= 5)
        dist_AcB_reps = ...
            S_star_partial_dist(A_partial_num_blocks_ne, ...
                    partial_dist_mat_AcB);
    elseif (num_B_reps_ne > 5)
        dist_AcB_reps = ...
            S_star_distance_reps_split(A_partial_num_blocks_ne, ...
                    partial_dist_mat_AcB);
    end
end