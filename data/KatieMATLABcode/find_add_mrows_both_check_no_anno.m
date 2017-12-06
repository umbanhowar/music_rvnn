function [add_rows] = ...
    find_add_mrows_both_check_no_anno(lst_no_anno, check_start_inds, k)

% FIND_ADD_MROWS_BOTH_CHECK_NO_ANNO finds diagonals of length K that
% neither start nor end at the same time steps as previously found repeats
% of length K.
% 
% INPUT: LST_NO_ANNO -- List of pairs of repeats
%        CHECK_INDS -- List of ending indices for repeats of length K
%                      that we use to check LST_NO_ANNO for more repeats of
%                      length K
%        K -- length of repeats that we are looking for 
%
% OUTPUT: ADD_ROWS -- List of newly found pairs of repeats of length K
%                     that are contained in larger repeats in LST_NO_ANNO

% Set-up the temporary and output variables
L = lst_no_anno;
knds = (L(:,5) == k);
search_inds = (L(:,5) > k);
add_rows = [];

% Multiply by SEARCH_INDS to narrow search to pairs of repeats of length
% greater than K
SI = L(:,1).*search_inds;
SJ = L(:,3).*search_inds;
EI = L(:,2).*search_inds;
EJ = L(:,4).*search_inds;

% Loop over CHECK_INDS 
for i = 1:length(check_start_inds)
    ci = check_start_inds(i);
    
    % Left Check: Check for CI on the left side of the pairs
    lnds = ((SI < ci) + (EI > (ci + k - 1)) == 2); 
    % Check that SI < CI and that EI > (CI + K - 1) indicating that there
    % is a repeat of length K with starting index CI contained in a larger
    % repeat which is the left repeat of a pair
    if sum(lnds) > 0
        SJ_li = L(lnds,3);
        EJ_li = L(lnds,4);
        l_num = size(SJ_li,1);
        
        % Left side of left pair
        l_left_k = ci*ones(l_num,1) - L(lnds,1); 
                   % Note: E-S+1 = (CI - 1) - L(LNDS,1) + 1
        l_add_left = [L(lnds,1), (ci - 1)*ones(l_num,1), ...
            SJ_li, (SJ_li + l_left_k - ones(l_num,1)), l_left_k];
        
        % Middle of left pair
        % Note: L_MID_K = K
        l_add_mid = [ci*ones(l_num,1), (ci + k - 1)*ones(l_num,1), ...
            (SJ_li + l_left_k), ...
            (SJ_li + l_left_k + (k - 1)*ones(l_num,1)), k*ones(l_num,1)];
        
        % Right side of left pair
        l_right_k = L(lnds,2) - ((ci + k) - 1)*ones(l_num,1);
        l_add_right = [(ci + k)*ones(l_num,1), L(lnds,2), ...
            (EJ_li - l_right_k + ones(l_num,1)), EJ_li, l_right_k];
        
        % Add the found rows
        add_rows = [add_rows; l_add_left; l_add_mid; l_add_right];
    end
    
    % Right Check: Check for CI on the right side of the pairs
    rnds = ((SJ < ci) + (EJ > (ci + k - 1)) == 2); 
    % Check that SI < CI and that EI > (CI + K - 1) indicating that there
    % is a repeat of length K with starting index CI contained in a larger
    % repeat which is the right repeat of a pair
    if sum(rnds) > 0
        SI_ri = L(rnds,1);
        EI_ri = L(rnds,2);
        r_num = size(SI_ri,1);
        
        % Left side of right pair
        r_left_k = ci*ones(r_num,1) - L(rnds,3); 
                   % Note: E-S+1 = (CI - 1) - L(LNDS,3) + 1
        r_add_left = [SI_ri, (SI_ri + r_left_k - ones(r_num,1)), ...
            L(rnds,3), (ci - 1)*ones(r_num,1), r_left_k];
        
        % Middle of right pair
        % Note: R_MID_K = K
        r_add_mid = ...
            [(SI_ri + r_left_k), (SI_ri +r_left_k + (k - 1)*ones(r_num,1)), ...
            ci*ones(r_num,1), (ci + k - 1)*ones(r_num,1), k*ones(r_num,1)];
        
        % Right side of right pair
        r_right_k = L(rnds,4) - ((ci + k) - 1)*ones(r_num,1);
        r_add_right = [(EI_ri - r_right_k + ones(r_num,1)), EI_ri, ...
            (ci + k)*ones(r_num,1), L(rnds,4), r_right_k];
        
        % Add the found rows
        add_rows = [add_rows; r_add_left; r_add_mid; r_add_right];
    end
    
end

end