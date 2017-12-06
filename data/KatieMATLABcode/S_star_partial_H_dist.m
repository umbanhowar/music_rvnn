function [partial_dist_LcR] =...
    S_star_partial_H_dist(L_num_blocks, partial_distmat_LcR)

% S_STAR_PARTIAL_H_DIST uses L_NUM_BLOCKS and PARTIAL_DISTMAT_LCR to
% compute the distance between two matrices L and R used to create
% PARTIAL_DISTMAT_LCR and L_NUM_BLOCKS. For this function, we assume that L
% has to have at least the same number of rows as R.
%
% INPUT: L_NUM_BLOCKS -- Number of repeats in each row of matrix L
%        PARTIAL_DISTMAT_LCR -- Pairwise distances between rows of L and
%                               rows of R
%
% OUTPUT: PARTIAL_DIST_LCR -- Distance between L and R

% Note: S_STAR_PARTIAL_H_DIST is very similar to S_STAR_PARTIAL_DIST

num_L_rows = size(partial_distmat_LcR,1);
num_R_rows = size(partial_distmat_LcR,2);

% Determine which rows in L are going to be compared to rows in R
zero_pad = num_L_rows - num_R_rows;

if zero_pad == 0
    % Do the permutation of the rows in R in a straight forward manner
    inds_mat = perms([1:num_R_rows]);
    num_perms = size(inds_mat,1);
    
    % In this case, NUM_BLOCK_MAT will be empty because we have a straight 
    %       comparison of rows in L to those in R
    num_block_mat = [];
    
else
    % First choose rows in L that the rows in R will compare to, then 
    %       permute the rows in R
    non_zero_inds = nchoosek([1:num_L_rows], num_R_rows);
    num_NZI = size(non_zero_inds,1);
    
    % Using NON_ZERO_INDS, build a mask for th evariable
    %       NUM_BLOCK_MAT_ONCE. For each row of NON_ZERO_INDS, the indices
    %       in the corresponding row of NUM_BLOCK_MASK are set to zero.
    num_block_mask = ones(num_L_rows,num_NZI);
    for perm_num = 1:num_NZI
        inds2zero = non_zero_inds(perm_num,:);
        num_block_mask(inds2zero,perm_num) = zeros(num_R_rows,1);
    end
    
    num_block_mat_once = repmat(L_num_blocks,1,num_NZI).*num_block_mask;
    
    % Permute rows in R
    perms_non_zero_inds = perms([1:num_R_rows]);
    num_H_perms = size(perms_non_zero_inds,1);
    
    % We desire to compare all possible permutations of rows in R to rows
    %       in L. So we found all the non-zero comparisons, but if we did
    %       not permute rows in R, we would only have been comparing rows
    %       in L (in order) to all combinations of rows in L. To make sure
    %       that we are doing all of the comparisons, we will loop over all
    %       the permutations of rows in R and list the columns of
    %       NON_ZERO_INDS in the order prescribed by each permutation of
    %       the rows in R.
    inds_mat = zeros(num_NZI*num_H_perms,num_R_rows); % Preallocate
    for rpi = 1:num_H_perms
        pNZI_row = perms_non_zero_inds(rpi,:);
        % Here we are stacking NON_ZERO_INDS in the order
        %       prescribed by each row of PERMS_NON_ZERO_INDS
        inds_mat(((rpi-1)*num_NZI + 1):(rpi*num_NZI), :) ...
            = non_zero_inds(:,pNZI_row);
    end
    num_perms = size(inds_mat,1);
    
    % Now we need to create the full matrix NUM_BLOCK_MAT. To create
    %       INDS_MAT, we looped over the number of permutations of the
    %       partial representations of B_REPS and simply copied the
    %       columns of NON_ZERO_INDS. In the same manner to create
    %       NUM_BLOCK_MAT, we just need to lay NUM_REP_PERMS copies of
    %       NUM_BLOCK_MAT_ONCE next to each other
    
    num_block_mat = repmat(num_block_mat_once,1,num_H_perms);
end

% In this next phase, we are creating a matrix that is comprised of two
%       parts: PARTIAL_DISTMAT_LCR as a column laid next to itself
%       NUM_PERMS times and then NUM_BLOCK_MAT under those columns. This
%       new matrix is a bit unintuitive. PARTIAL_DISTMAT_LCR encodes the
%       distances between the rows in L and those in R. Turning
%       PARTIAL_DISTMAT_LCR into a column, we have rows of L compared to
%       the first row in R, then compared to the second row in R, and then
%       the third and so on. Since we have already computed all these
%       distances, what we need to do is create a mask that allows for only
%       NUM_R_ROWS of these distances to be in each column. But more than 
%       that, we need to ensure that we are comparing rows in R to 
%       NUM_R_ROWS distinct rows in L.

% Create list of indices for comparing rows in L to rows in R as described
%       above. In the below line, we create a row that will force the
%       i-th column of INDS_MAT to be between (I-1)*NUM_L_ROWS and
%       I*NUM_L_ROWS as desired.
add_row = [0:num_L_rows:(num_L_rows*(num_R_rows - 1))];
inds_mat = inds_mat + repmat(add_row,num_perms,1);

% Create a mask for the column matrix of PARTIAL_DIST_ACB described
%       above. The mask will be 1 when we want to use the computed
%       distances in that particular column (which represents one
%       particular comparison of rows from L and those from R).
mask_comp_mat = zeros(num_L_rows*num_R_rows, num_perms);
for ind = 1:num_perms
    row_inds = inds_mat(ind,:);
    mask_comp_mat(row_inds,ind) = ones(num_R_rows,1);
end

% Create COMP_MAT by multiplying MASK_COMP_MAT element-wise with
%       PARTIAL_DISTMAT_LCR as a column laid next to itself NUM_PERMS 
%       times and then adding NUM_BLOCK to the bottom of the result of that
%       element-wise multiplication.
comp_mat = repmat(partial_distmat_LcR(:),1,num_perms).*mask_comp_mat;
comp_mat = [comp_mat;num_block_mat];

% Finally, find the distance between A_REPS and B_REPS
partial_dist_LcR = min(sum(comp_mat,1));

% Clear all but output
clear L_num_blocks; clear partial_distmat_LcR; clear num_L_rows; 
clear num_R_rows; clear zero_pad; clear inds_mat; clear num_perms;
clear num_block_mat; clear non_zero_inds; clear num_NZI; 
clear num_block_mask; clear inds2zero; clear num_block_mat_once; 
clear perms_non_zero_inds; clear num_H_perms; clear pNZI_row;
clear add_row; clear mask_comp_mat; clear row_inds; clear comp_mat;
clear ind; clear perm_num; clear rpi;

end