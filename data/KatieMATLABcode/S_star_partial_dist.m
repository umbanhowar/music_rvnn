function [partial_rep_dist_AcB] =...
    S_star_partial_dist(A_num_blocks, partial_distmat_AcB)

% S_STAR_PARTIAL_H_DIST uses A_NUM_BLOCKS and PARTIAL_DISTMAT_ACB to
% compute the distance two lists of partial representations A_PARTIAL_REPS
% and B_PARTIAL_REPS used to create PARTIAL_DISTMAT_ACB and A_NUM_BLOCKS.
% We assume that A_PARTIAL_REPS has at least the same number of partial
% representations as B_PARTIAL_REPS.
%
% INPUT: A_PARTIAL_NUM_BLOCKS -- Number of repeats in each partial
%                                representation in A_PARTIAL_REPS
%        PARTIAL_DISTMAT_ACB -- Pairwise distances between the partial
%                               representations in A_PARTIAL_REPS and
%                               those in B_PARTIAL_REPS
%
% OUTPUT: PARTIAL_REP_DIST_ACB -- Distance between two lists of partial
%                                 representations A_PARTIAL_REPS and
%                                 B_PARTIAL_REPS

% Note: S_STAR_PARTIAL_DIST is very similar to S_STAR_PARTIAL_H_DIST

num_A_reps = size(partial_distmat_AcB,1);
num_B_reps = size(partial_distmat_AcB,2);

% Determine which partial representations in A_PARTIAL_REPS are going to be 
%       compared the partial representations in B_PARTIAL_REPS
zero_pad = num_A_reps - num_B_reps;

if zero_pad == 0
    % Do the permutation of the partial representations in B_PARTIAL_REPS
    %       in a straight forward manner
    inds_mat = perms([1:num_B_reps]);
    num_perms = size(inds_mat,1);
    
    % In this case, NUM_BLOCK_MAT will be empty because we have a straight 
    %       comparison of the partial representations of A_PARTIAL_REPS and
    %       those in B_PARTIAL_REPS.
    num_block_mat = [];
    
else
    % First choose the partial representations in A_PARTIAL_REPS that the
    %       partial representations in B_PARTIAL_REPS will compare to, then
    %       permute the partial representations in B_PARTIAL_REPS
    non_zero_rep_inds = nchoosek([1:num_A_reps], num_B_reps);
    num_NZRI = size(non_zero_rep_inds,1);
    
    % Now using NON_ZERO_REP_INDS, we build a mask for the variable
    %       NUM_BLOCK_MAT_ONCE. For each row of NON_ZERO_REP_INDS, the
    %       indices in the corresponding row of NUM_BLOCK_MASK are set to
    %       zero.
    num_block_mask = ones(num_A_reps,num_NZRI);
    for perm_num = 1:num_NZRI
        inds2zero = non_zero_rep_inds(perm_num,:);
        num_block_mask(inds2zero,perm_num) = zeros(num_B_reps,1);
    end
    
    num_block_mat_once = repmat(A_num_blocks',1,num_NZRI).*num_block_mask;
    
    % Permute the partial representations in B_PARTIAL_REPS
    perms_non_zero_rep_inds = perms([1:num_B_reps]);
    num_rep_perms = size(perms_non_zero_rep_inds,1);
    
    % We desire to compare all possible permutations of partial
    %       representations in B_PARTIAL_REPS to the partial
    %       representations in A_PARTIAL_REPS. So we found all the non-zero
    %       comparisons, but if did not permute the partial representations
    %       in B_PARTIAL_REPS, we would only have been comparing the
    %       partial representations in A_PARTIAL_REPS in order to all
    %       combinations of the partial representations in A_PARTIAL_REPS.
    %       To make sure that we are doing all of the comparisons, we will
    %       loop over all the permutations of partial representations in
    %       B_PARTIAL_REPS and list the colums of NON_ZERO_REP_INDS in the
    %       order prescribed by each permutation of the partial
    %       representations in B_PARTIAL_REPS.
    inds_mat = zeros(num_NZRI*num_rep_perms,num_B_reps); % Preallocate
    for rpi = 1:num_rep_perms
        pNZRI_row = perms_non_zero_rep_inds(rpi,:);
        % Stack NON_ZERO_REP_INDS in the order prescribed by each row of 
        %       PERMS_NON_ZERO_REP_INDS
        inds_mat(((rpi-1)*num_NZRI + 1):(rpi*num_NZRI), :) ...
            = non_zero_rep_inds(:,pNZRI_row);
    end
    num_perms = size(inds_mat,1);
    
    % Now we need to create the full matrix NUM_BLOCK_MAT. To create
    %       INDS_MAT, we looped over the number of permutations of the
    %       partial representations of B_PARTIAL_REPS and simply copied the
    %       columns of NON_ZERO_REP_INDS. In the same manner to create
    %       NUM_BLOCK_MAT, we just lay NUM_REP_PERMS copies of
    %       NUM_BLOCK_MAT_ONCE next to each other
    
    num_block_mat = repmat(num_block_mat_once,1,num_rep_perms);
end

% In this next phase, we are creating a matrix that is comprised of two
%       parts: PARTIAL_DIST_ACB as a column laid next to itself NUM_PERMS
%       times and then NUM_BLOCK_MAT under those columns. This new matrix
%       is a bit unintuitive. PARTIAL_DIST_ACB encodes the distances
%       between the partial representations in A_PARTIAL_REPS and those in
%       B_PARTIAL_REPS. Turning PARTIAL_DIST_ACB into a column, we have
%       partial representations of A_PARTIAL_REPS compared to the first
%       partial representation in B_PARTIAL_REPS, then compared to the
%       second partial representation in B_PARTIAL_REPS, and then the third
%       and so on. Since we have already computed all these distances, what
%       we need to do is create a mask that allows for only NUM_B_REPS of
%       these distances to be in each column. But more than that we need to
%       ensure that we are comparing the partial representations in
%       B_PARTIAL_REPS to NUM_B_REPS distinct partial representations in
%       A_PARTIAL_REPS

% Create list of indices for comparing the partial representatiosn in
%       A_PARTIAL_REPS to the partial representations in B_REPS as
%       described above. In the below line, we create row that will force
%       the i-th column of INDS_MAT to be between (I-1)*NUM_A_REPS and
%       I*NUM_A_REPS as desired.
add_row = [0:num_A_reps:(num_A_reps*(num_B_reps - 1))];
inds_mat = inds_mat + repmat(add_row,num_perms,1);

% Create a mask for the column matrix of PARTIAL_DIST_ACB described
%       above. Here the mask will be one when we want to use the
%       computed distances in that particular column (which represents
%       one particular comparison of partial representations from
%       A_PARTIAL_REPS and those from B_PARTIAL_REPS.
mask_comp_mat = zeros(num_A_reps*num_B_reps, num_perms);
for ind = 1:num_perms
    row_inds = inds_mat(ind,:);
    mask_comp_mat(row_inds,ind) = ones(num_B_reps,1);
end

% We get COMP_MAT by mutliplying MASK_COMP_MAT element-wise with
%       PARTIAL_DIST_ACB as a column laid next to itself NUM_PERMS times
%       and then adding num_block to the bottom of the result of that
%       element-wise multiplication.
comp_mat = repmat(partial_distmat_AcB(:),1,num_perms).*mask_comp_mat;
comp_mat = [comp_mat;num_block_mat];

% Finally, find the distance between A_PARTIAL_REPS and B_PARTIAL_REPS
partial_rep_dist_AcB = min(sum(comp_mat,1));

% Clear all but output
clear A_num_blocks; clear partial_dist_AcB; clear num_A_reps; 
clear num_B_reps; clear zero_pad; clear inds_mat; clear num_perms;
clear num_block_mat; clear non_zero_rep_inds; clear num_NZRI; 
clear num_block_mask; clear inds2zero; clear num_block_mat_once; 
clear perms_non_zero_rep_inds; clear num_rep_perms; clear pNZRI_row;
clear add_row; clear mask_comp_mat; clear row_inds; clear comp_mat;
clear ind; clear perm_num; clear rpi;

end