function [add_rows] = ...
    find_add_erows_both_check_no_anno(lst_no_anno, check_inds, k)

% FIND_ADD_EROWS_BOTH_CHECK_NO_ANNO finds diagonals of length K that end at
% the same time step as previously found repeats of length K. 
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
EI = L(:,2).*search_inds;
EJ = L(:,4).*search_inds;

% Loop over CHECK_INDS
for i = 1:length(check_inds)
    ci = check_inds(i);
    
    % Left Check: Check for CI on the left side of the pairs
    lnds = (EI == ci); % Check if the end index of the left repeat of the
                       % pair equals CI
    if sum(lnds) > 0
        EJ_li = L(lnds,4);
        l_num = size(EJ_li,1);
        l_add = [(L(lnds,2) - k + 1), L(lnds,2), ...
            (EJ_li - k + 1), EJ_li, k*ones(l_num,1)];
        l_add_left = [L(lnds,1), (L(lnds,2) - k), ...
            L(lnds,3), (EJ_li - k), (L(lnds,5) - k)];
        % Add the found pairs of repeats
        add_rows = [add_rows; l_add; l_add_left];
    end
    
    % Right Check: Check for CI on the right side of the pairs
    rnds = (EJ == ci); % Check if the end index of the right repeat of the
                       % pair equals CI
    if sum(rnds) > 0
        EI_ri = L(rnds,2);
        r_num = size(EI_ri,1);
        r_add = [(EI_ri - k + 1), EI_ri, ...
            (L(rnds,4) - k + 1), L(rnds,4), k*ones(r_num,1)];
        r_add_left = [L(rnds,1), (EI_ri - k), ...
            L(rnds,3), (L(rnds,4) - k), (L(rnds,5) - k)];
        % Add the found pairs of repeats
        add_rows = [add_rows; r_add];
    end
    
end

end