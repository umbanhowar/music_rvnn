function rep_dist_AB = dist_between_reps(rep_A, key_A, rep_B, key_B)

% DIST_BETWEEN_REPS computes the distance between two representations,
% denoted here as REP_A and REP_B. Associated to these two representations
% are KEY_A and KEY_B.
%
% INPUT: REP_A -- Binary matrix of one representation encoding where
%                 repeats begin
%        KEY_A -- Vector of lengths for the repeats encoded in REP_A
%        REP_B -- Binary matrix of one representation encoding where
%                 repeats begin
%        KEY_B -- Vector of lengths for the repeats encoded in REP_B
%
% OUTPUT: REP_DIST_AB -- Distance between two representations REP_A (with
%                        KEY_A) and REP_B (with KEY_B)

% Step 0 - Check that KEY_A and KEY_B are column vectors. If they are not,
%          then make them column vectors.

if min(size(key_A)) == 1 && min(size(key_B)) == 1
    if size(key_A,2) > 1
        key_A = key_A';
    end
    if size(key_B,2) > 1
        key_B = key_B';
    end 
else
    fprintf(['ERROR: at least one of the representation keys ', ...
        'is not a vector \n']);
end

% Step 1 - Find the intersection of KEY_A and KEY_B, call this intersection
%          KEY_COMPARE

key_compare = intersect(key_A, key_B);

if isempty(key_compare)
    rep_dist_AB = sum(sum(rep_A)) + sum(sum(rep_B));
else
    
    L_key_compare = size(key_compare,1);
    
    % Step 2 - Using KEY_COMPARE with KEY_A and KEY_B, find:
    %          (a) the number of blocks that exist in REP_A and REP_B that
    %              correspond to the bandwidths in KEY_A and KEY_B that 
    %              are not in KEY_COMPARE
    %          (b) the rows in REP_A and REP_B that need to be compared
    
    L_key_A = size(key_A,1);
    L_key_B = size(key_B,1);
    
    inds2sum_A = (sum(repmat(key_A, 1, L_key_compare) == ...
        repmat(key_compare', L_key_A, 1),2) == 0);
    
    inds2sum_B = (sum(repmat(key_B, 1, L_key_compare) == ...
        repmat(key_compare', L_key_B, 1),2) == 0);
    
    num_blocks_with_no_comparison = ...
        sum(sum(rep_A(inds2sum_A,:))) + sum(sum(rep_B(inds2sum_B,:)));
    
    inds2check_A = (sum(repmat(key_A, 1, L_key_compare) == ...
        repmat(key_compare', L_key_A, 1),2) == 1);
    
    inds2check_B = (sum(repmat(key_B, 1, L_key_compare) == ...
        repmat(key_compare', L_key_B, 1),2) == 1);
    
    rep_Ac = rep_A(inds2check_A,:);
    key_Ac = key_A(inds2check_A);
    
    rep_Bc = rep_B(inds2check_B,:);
    key_Bc = key_B(inds2check_B);
    
    % Step 3 - Loop over the bandwidths in KEY_COMPARE, computing the 
    %          distance between the rows in REP_A and REP_B that encode the
    %          same size repetitions and add all these distances together
    
    dist_AcB = 0;
    
    for ki = 1:L_key_compare
        k = key_compare(ki);
        smallA_ind = find(key_Ac > k,1) - 1;
        if isempty(smallA_ind)
            smallA_ind = size(key_Ac,1);
        end
        smallA = rep_Ac(1:smallA_ind,:);
        
        smallB_ind = find(key_Bc > k,1) - 1;
        if isempty(smallB_ind)
            smallB_ind = size(key_Bc,1);
        end
        smallB = rep_Bc(1:smallB_ind,:);
        
        % The distance between SMALLA and SMALLB is the same as the
        % distance between SMALLB and SMALLA, but due to how the code is
        % structured, we put SMALLA first in the computation if it has as
        % many or more rows than SMALLB. If SMALLB has more rows than
        % SMALLA, then we put SMALLB first. In summary: 
        %       If SMALLA_IND >= SMALLB_IND, compute dist(A,B). 
        %       If SMALLA_IND < SMALLB_IND, compute dist(B,A).
        if smallA_ind >= smallB_ind
            % SMALLA has as many or more rows as SMALLB
            dist_AcB = dist_AcB + S_star_distance(smallA,smallB);
        else
            % SMALLB has more rows than SMALLA
            dist_AcB = dist_AcB + S_star_distance(smallB,smallA);
        end
        
        % Remove compared rows from consideration
        rep_Ac(1:smallA_ind,:) = [];
        rep_Bc(1:smallB_ind,:) = [];
        
        key_Ac(1:smallA_ind,:) = [];
        key_Bc(1:smallB_ind,:) = [];
        
    end
    
    rep_dist_AB = dist_AcB + num_blocks_with_no_comparison;
    
end