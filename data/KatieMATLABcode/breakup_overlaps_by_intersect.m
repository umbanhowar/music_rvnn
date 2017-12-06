function [pattern_no_overlaps, pattern_no_overlaps_key] = ...
    breakup_overlaps_by_intersect(input_pattern_obj, bw_vec, thresh_bw)

% BREAKUP_OVERLAPS_BY_INTERSECT distills repeats encoded in
% INPUT_PATTERN_OBJ (and BW_VEC) to the essential structure components, the
% set of repeats so that no time step is contained in more than one repeat.
%
% INPUT: INPUT_PATTERN_OBJ -- Binary matrix with 1's where repeats begin 
%                             and 0's otherwise
%        BW_VEC -- Vector containing the lengths of the repeats encoded in
%                  INPUT_PATTERN_OBJ
%        THRESH_BW -- The smallest allowable repeat length
%
% OUTPUT: PATTERN_NO_OVERLAPS -- Binary matrix with 1's where repeats of
%                                essential structure components begin
%         PATTERN_NO_OVERLAPS_KEY -- Vector containing the lengths of the 
%                                    repeats of essential structure
%                                    components encoded in
%                                    PATTERN_NO_OVERLAPS

% Initialize the inputs
if nargin < 3
    T = 0;
else
    T = thresh_bw;
end

PNO = input_pattern_obj;
sn = size(PNO,2);

% Sort the BW_VEC and the PNO so that we process the biggest pieces first
[bw_vec, bnds] = sort(bw_vec, 'descend');
PNO = PNO(bnds,:);
T_ind = find(bw_vec == T,1) - 1;
if isempty(T_ind)
    T_ind = length(bw_vec);
end

[PNO_block] = reconstruct_full_block(PNO, bw_vec);

% Check stopping condition -- Are there overlaps?
while sum(sum(PNO_block(1:T_ind,:),1) > 1) > 0
    % Find all overlaps by comparing the rows of repeats pairwise
    [overlaps_PNO_block] = check_overlaps(PNO_block);
    
    % Remove the rows with bandwidth T or less from consideration
    overlaps_PNO_block((T_ind+1):end,:) = 0;
    overlaps_PNO_block(:,(T_ind+1):end) = 0;
    
    % Find the first two groups of repeats that overlap, calling one group
    % RED and the other group BLUE
    [ri,bi] = find(overlaps_PNO_block,1);
    
    red = PNO(ri,:);
    RL = bw_vec(ri,:);
    
    blue = PNO(bi,:);
    BL = bw_vec(bi,:);
    
    % Compare the repeats in RED and BLUE, cutting the repeats in those
    % groups into non-overlapping pieces
    [union_mat, union_length] = compare_and_cut(red, RL, blue, BL);
    
    PNO([ri,bi],:) = [];
    bw_vec([ri,bi],:) = [];
    
    PNO = [PNO; union_mat];
    bw_vec = [bw_vec; union_length];
    
    % Check there are any repeats of length 1 that should be merged into
    % other groups of repeats of length 1 and merge them if necessary
    if sum(union_length == 1) > 0
        [PNO, bw_vec] = merge_based_on_length(PNO, bw_vec, 1);
    end
    
    % Again sort the BW_VEC and the PNO so that we process the biggest
    % pieces first
    [bw_vec, bnds] = sort(bw_vec, 'descend');
    PNO = PNO(bnds,:);
    
    % Find the first row that contains repeats of length less than T and
    % remove these rows from consideration during the next check of the
    % stopping condition
    T_ind = find(bw_vec == T,1) - 1;
    if isempty(T_ind)
        T_ind = length(bw_vec);
    end
    
    [PNO_block] = reconstruct_full_block(PNO, bw_vec);
end

[bw_vec, bnds] = sort(bw_vec, 'ascend');
pattern_no_overlaps = PNO(bnds,:);
pattern_no_overlaps_key = bw_vec;

end