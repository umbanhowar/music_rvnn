function [lst_out] = find_complete_list(pair_lst, sn)

% FIND_COMPLETE_LIST finds all smaller diagonals (and the associated pairs
% of repeats) that are contained in larger diagonals found previously.
%
% INPUT: PAIR_LST -- List of pairs of repeats found in earlier step
%                    (bandwidths MUST be in ascending order). If you have
%                    run LIGHTUP_LST_WITH_THRESH_BW before this script,
%                    then PAIR_LIST will be ordered correctly. 
%        SN -- Song length, which is the number of audio shingles
%
% OUTPUT: LST_OUT -- List of pairs of repeats with smaller repeats added
%                    and with annotation markers

% Find the list of unique repeat lengths
bfound = unique(pair_lst(:,5));
b = length(bfound); % Number of unique repeat lengths

% If the longest bandwidth is the length of the song, then remove that row
if sn == bfound(b)
    pair_lst(end,:) = [];
    bfound(end) = [];
    b = (b - 1);
end
p = size(pair_lst,1);

% Initialize temp variables
add_mat = [];

% Step 1: For each found bandwidth, search upwards (i.e. search the larger 
%         bandwidths) and add all found diagonals to the variable ADD_MAT
for i = 1:b
    % Set the bandwidth based on BFOUND
    bw = bfound(i); 
    
    % Isolate pairs of repeats that are length BW
    bsnds = find(pair_lst(:,5) == bw, 1);
    bends = find(pair_lst(:,5) > bw, 1) - 1; % This is ok since [] - 1 = []
    if isempty(bends)
        bends = p;
    end
    
    % Part A1: Isolate all starting time steps of the repeats of length BW
    SI = pair_lst(bsnds:bends,1);
    SJ = pair_lst(bsnds:bends,3);
    all_vec_snds = [SI;SJ];
    int_snds = unique(all_vec_snds);
    
    % Part A2: Isolate all ending time steps of the repeats of length BW
    EI = pair_lst(bsnds:bends,2); % Similar to definition for SI
    EJ = pair_lst(bsnds:bends,4); % Similar to definition for SJ
    all_vec_ends = [EI;EJ];
    int_ends = unique(all_vec_ends);
    
    % Part B: Use the current diagonal information to search for diagonals 
    %         of length BW contained in larger diagonals and thus were not
    %         detected because they were contained in larger diagonals that
    %         were removed by our method of eliminating diagonals in
    %         descending order by size
    [add_srows] = find_add_srows_both_check_no_anno(pair_lst, int_snds, bw);
    [add_mrows] = find_add_mrows_both_check_no_anno(pair_lst, int_snds, bw);
    [add_erows] = find_add_erows_both_check_no_anno(pair_lst, int_ends, bw);

    % Add the new pairs of repeats to the temporary list ADD_MAT
    add_mat = [add_mat; add_srows; add_mrows; add_erows];
end

% Step 2: Combine PAIR_LST and ADD_MAT. Make sure that you don't have any
%         double rows in ADD_MAT. Then find the new list of found 
%         bandwidths in COMBINE_MAT
combine_mat = [pair_lst; add_mat];
combine_mat = unique(combine_mat, 'rows');
[~,combine_inds] = sort(combine_mat(:,5));
combine_mat = combine_mat(combine_inds,:);
c = size(combine_mat,1);

% Again, find the list of unique repeat lengths
new_bfound = unique(combine_mat(:,5));
new_b = length(new_bfound);

full_lst = [];
% Step 3: Loop over the new list of found bandwidths to add the annotation
%         markers to each found pair of repeats
for j = 1:new_b
    new_bw = new_bfound(j);
    % Isolate pairs of repeats in COMBINE_MAT that are length BW
    new_bsnds = find(combine_mat(:,5) == new_bw, 1);
    new_bends = find(combine_mat(:,5) > new_bw, 1) - 1;
                % This is ok since [] - 1 = []
    if isempty(new_bends)
        new_bends = c;
    end
    
    bw_mat = [combine_mat(new_bsnds:new_bends,:)];
    length_bw_mat = size(bw_mat,1);
    
    temp_anno_lst = [bw_mat, zeros(length_bw_mat,1)];
    % Part C: Get annotation markers for this bandwidth
    [temp_anno_lst] = add_annotations(temp_anno_lst, sn);
    full_lst = [full_lst; temp_anno_lst]; 
end

lst_out = full_lst;

end