function dist_between_partialsAB = dist_two_partial_sets(A_partial_reps, ...
    A_partial_keys, A_partial_widths, A_partial_num_blocks, ...
    B_partial_reps, B_partial_keys, B_partial_widths, B_partial_num_blocks)

% DIST_TWO_PARTIAL_SETS computes the distance between two lists of partial
% representations, denoted here as A_PARTIAL_REPS and B_PARTIAL_REPS.
% Associated to A_PARTIAL_REPS are A_PARTIAL_KEYS, A_PARTIAL_WIDTHS, and
% A_PARTIAL_NUM_BLOCKS. Associated to B_PARTIAL_REPS are B_PARTIAL_KEYS,
% B_PARTIAL_WIDTHS, and B_PARTIAL_NUM_BLOCKS.
%
% INPUT: A_PARTIAL_REPS -- First list of partial hierarchical
%                          representations
%        A_PARTIAL_KEYS -- List of vectors of lengths of repeats in the
%                          partial representations. Each vector of lengths
%                          corresponds to a partial hierarchical
%                          representation stored in A_PARTIAL_REPS
%        A_PARTIAL_WIDTHS -- List of the widths corresponding to the
%                            partial hierarchical representations stored
%                            in A_PARTIAL_REPS
%        A_PARTIAL_NUM_BLOCKS -- Number of repeats in each partial
%                                representation in A_PARTIAL_REPS
%        B_PARTIAL_REPS -- Second list of partial hierarchical 
%                          representations
%        B_PARTIAL_KEYS -- List of vectors of lengths of repeats in the
%                          partial representations. Each vector of lengths
%                          corresponds to a partial hierarchical
%                          representation stored in B_PARTIAL_REPS
%        B_PARTIAL_WIDTHS -- List of the widths corresponding to the
%                            partial hierarchical representations stored
%                            in B_PARTIAL_REPS
%        B_PARTIAL_NUM_BLOCKS -- Number of repeats in each partial
%                                representation in B_PARTIAL_REPS
%
% OUTPUT: DIST_BETWEEN_PARTIALSAB -- Distance between two lists of partial
%                                    representations A_PARTIAL_REPS and
%                                    B_PARTIAL_REPS


% Step 0 - Check that A_PARTIAL_WIDTHS and B_PARTIAL_WIDTHS are column
%          vectors. If they are not, then make them column vectors.

if min(size(A_partial_widths)) == 1 && min(size(B_partial_widths)) == 1
    if size(A_partial_widths,2) > 1
        A_partial_widths = A_partial_widths';
    end
    if size(B_partial_widths,2) > 1
        B_partial_widths = B_partial_widths';
    end 
else
    fprintf(['ERROR: at least one of the partial representation ', ...
        'width keys is not a vector \n']);
end

% Step 1 - Find the intersection of A_PARTIAL_WIDTHS and B_PARTIAL_WIDTHS, 
%          call this intersection WIDTH_COMPARE.

width_compare = intersect(A_partial_widths, B_partial_widths);

if isempty(width_compare)
    dist_between_partialsAB = ...
        sum(A_partial_num_blocks) + sum(B_partial_num_blocks);
else
    
    L_width_compare = size(width_compare,1);
    
    % Step 2 - Using WIDTH_COMPARE with A_PARTIAL_WIDTHS and 
    %          B_PARTIAL_WIDTHS, find:
    %          (a) the number of blocks that exist in the partial 
    %              representations in A_PARTIAL_REPS and B_PARTIAL_REPS 
    %              that correspond to the widths in A_PARTIAL_WIDTHS and  
    %              B_PARTIAL_WIDTHS that are NOT in WIDTH_COMPARE
    %          (b) the partial representations in A_PARTIAL_REPS and 
    %              B_PARTIAL_REPS that need to be compared
    
    % Step 2(a): Find the number of blocks that exist in the partial 
    %            representations in A_PARTIAL_REPS and B_PARTIAL_REPS 
    %            that correspond to the widths in A_PARTIAL_WIDTHS and  
    %            B_PARTIAL_WIDTHS that are NOT in WIDTH_COMPARE
    A_num_partial_reps = size(A_partial_widths,1);
    B_num_partial_reps = size(B_partial_widths,1);
    
    inds2sum_A = (sum(repmat(A_partial_widths, 1, L_width_compare) == ...
        repmat(width_compare', A_num_partial_reps, 1),2) == 0);
    
    inds2sum_B = (sum(repmat(B_partial_widths, 1, L_width_compare) == ...
        repmat(width_compare', B_num_partial_reps, 1),2) == 0);
    
    num_blocks_with_no_comparison = sum(A_partial_num_blocks(inds2sum_A)) ...
        + sum(B_partial_num_blocks(inds2sum_B));
    
    % Step 2(b): Find the partial representations in A_PARTIAL_REPS and 
    %            B_PARTIAL_REPS that need to be compared
    inds2check_A = (sum(repmat(A_partial_widths, 1, L_width_compare) == ...
        repmat(width_compare', A_num_partial_reps, 1),2) == 1);
    
    inds2check_B = (sum(repmat(B_partial_widths, 1, L_width_compare) == ...
        repmat(width_compare', B_num_partial_reps, 1),2) == 1);
    
    % Isolate the partial representations and their associated information
    %       that need to be compared
    Ac_partial_reps = A_partial_reps(inds2check_A);
    Ac_partial_keys = A_partial_keys(inds2check_A);
    Ac_partial_num_blocks = A_partial_num_blocks(inds2check_A);
    Ac_partial_widths = A_partial_widths(inds2check_A);
    
    Bc_partial_reps = B_partial_reps(inds2check_B);
    Bc_partial_keys = B_partial_keys(inds2check_B);
    Bc_partial_num_blocks = B_partial_num_blocks(inds2check_B);
    Bc_partial_widths = B_partial_widths(inds2check_B);
    
    % Step 3 - Loop over the bandwidths in WIDTH_COMPARE, computing the 
    %          distance between the partial representations in 
    %          A_PARTIAL_REPS and B_PARTIAL_REPS that are the same width.
    
    dist_AcB = 0;
    
    for wi = 1:L_width_compare
        w = width_compare(wi);
        
        % Isolate the partial representations that are of width W and their
        %       associated information 
        smallA_ind = find(Ac_partial_widths > w,1) - 1;
        if isempty(smallA_ind)
            smallA_ind = size(Ac_partial_widths,1);
        end
        smallA_partial_reps = Ac_partial_reps(1:smallA_ind);
        smallA_partial_keys = Ac_partial_keys(1:smallA_ind);
        smallA_partial_num_blocks = Ac_partial_num_blocks(1:smallA_ind);
        
        smallB_ind = find(Bc_partial_widths > w,1) - 1;
        if isempty(smallB_ind)
            smallB_ind = size(Bc_partial_widths,1);
        end
        smallB_partial_reps = Bc_partial_reps(1:smallB_ind);
        smallB_partial_keys = Bc_partial_keys(1:smallB_ind);
        smallB_partial_num_blocks = Bc_partial_num_blocks(1:smallB_ind);
        
        % Compute the distance between these isolated representations. The
        % distance between SMALLA and SMALLB is the same as the distance
        % between SMALLB and SMALLA, but due to how the code is structured,
        % we put SMALLA first in the computation if it has as many or more
        % rows than SMALLB. If SMALLB has more rows than SMALLA, then we
        % put SMALLB first. In summary: 
        %       If SMALLA_IND >= SMALLB_IND, compute dist(A,B). 
        %       If SMALLA_IND < SMALLB_IND, compute dist(B,A).
        if smallA_ind >= smallB_ind
            % SMALLA has as many or more rows as SMALLB
            dist2add = S_star_distance_reps(smallA_partial_reps, ...
                smallA_partial_keys, smallA_partial_num_blocks, ...
                smallB_partial_reps, smallB_partial_keys);
        elseif smallA_ind < smallB_ind
            % SMALLB has more rows than SMALLA
            dist2add = S_star_distance_reps(smallB_partial_reps, ...
                smallB_partial_keys, smallB_partial_num_blocks, ...
                smallA_partial_reps, smallA_partial_keys);
        end
        
        % Add the found distance between the isolated partial 
        %       representations in A_PARTIAL_REPS and B_PARTIAL_REPS to the 
        %       total distance between all the partial representations in 
        %       A_PARTIAL_REPS and B_PARTIAL_REPS
        dist_AcB = dist_AcB + dist2add;
        
        % Remove compared isolated representations from consideration:
        Ac_partial_reps(1:smallA_ind) = [];
        Ac_partial_keys(1:smallA_ind) = [];
        Ac_partial_num_blocks(1:smallA_ind) = [];
        Ac_partial_widths(1:smallA_ind) = [];
        
        Bc_partial_reps(1:smallB_ind) = [];
        Bc_partial_keys(1:smallB_ind) = [];
        Bc_partial_num_blocks(1:smallB_ind) = [];
        Bc_partial_widths(1:smallB_ind) = [];
    end
    
    dist_between_partialsAB = dist_AcB + num_blocks_with_no_comparison;
end