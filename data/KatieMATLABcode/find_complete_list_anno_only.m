function [lst_out] = find_complete_list_anno_only(pair_lst, sn)

% FIND_COMPLETE_LIST_ANNO_ONLY finds the annotations for all pairs of
% repeats found in a previous step.
%
% This code is similar to the second part of FIND_COMPLETE_LIST. 
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


full_lst = [];
% Loop over the list of found bandwidths to add the annotation markers to
% each found pair of repeats
for j = 1:b
    bw = bfound(j);
    % Isolate pairs of repeats in COMBINE_MAT that are length BW
    bsnds = find(pair_lst(:,5) == bw, 1);
    bends = find(pair_lst(:,5) > bw, 1) - 1;
                % This is ok since [] - 1 = []
    if isempty(bends)
        bends = p;
    end
    
    bw_mat = [pair_lst(bsnds:bends,:)];
    length_bw_mat = size(bw_mat,1);
    
    temp_anno_lst = [bw_mat, zeros(length_bw_mat,1)];
    % Get annotation markers for this bandwidth
    [temp_anno_lst] = add_annotations(temp_anno_lst, sn);
    full_lst = [full_lst; temp_anno_lst]; 
end

lst_out = full_lst;

end