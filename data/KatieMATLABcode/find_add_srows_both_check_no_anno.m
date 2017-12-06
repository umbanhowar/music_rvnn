function [add_rows] = ...
    find_add_srows_both_check_no_anno(lst_no_anno, check_inds, k)

% FIND_ADD_SROWS_BOTH_CHECK_NO_ANNO finds diagonals of length K that start 
% at the same time step as previously found repeats of length K. 
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

% Loop over CHECK_INDS 
for i = 1:length(check_inds)
    ci = check_inds(i);
    
    % Left Check: Check for CI on the left side of the pairs
    lnds = (SI == ci); % Check if the starting index of the left repeat of
                       % the pair equals CI
    if sum(lnds) > 0
        SJ_li = L(lnds,3);
        l_num = size(SJ_li,1);
        l_add = [L(lnds,1), (L(lnds,1) + k - 1), ...
            SJ_li, (SJ_li + k - 1), k*ones(l_num,1)];
        l_add_right = [(L(lnds,1) + k), L(lnds,2), ...
            (SJ_li + k), L(lnds,4), (L(lnds,5) - k)];
        % Add the found rows
        add_rows = [add_rows; l_add; l_add_right];
    end
    
    % Right Check: Check for CI on the right side of the pairs
    rnds = (SJ == ci); % Check if the starting index of the right repeat of
                       % the pair equals CI      
    if sum(rnds) > 0
        SI_ri = L(rnds,1);
        r_num = size(SI_ri,1);
        r_add = [SI_ri, (SI_ri + k - 1), ...
            L(rnds,3), (L(rnds,3) + k - 1), k*ones(r_num,1)];
        r_add_right = [(SI_ri + k), L(rnds,2), ...
            (L(rnds,3) + k), L(rnds,4), (L(rnds,5) - k)];
        % Add the found rows
        add_rows = [add_rows; r_add; r_add_right];
    end
    
end

end