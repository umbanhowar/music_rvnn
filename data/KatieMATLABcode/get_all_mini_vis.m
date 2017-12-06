function [all_partial_reps, all_partial_keys, all_partial_widths, ...
    all_partial_num_blocks] = get_all_mini_vis(full_vis, full_vis_key)

% GET_ALL_MINI_VIS uses FULL_VIS and FULL_VIS_KEY to get all the partial
% hierarchical representations for the hierarchical representation stored
% in FULL_MATRIX_NO (and FULL_VIS_KEY)
%
% INPUT: FULL_VIS -- Binary matrix with 1's where hierarchical structure 
%                    begins and 0's otherwise
%        FULL_VIS_KEY -- Vector containing the lengths of the hierarchical
%                        structure encoded in FULL_VIS
% 
% OUTPUT: ALL_PARTIAL_REPS -- List of partial hierarchical representations
%                             for the hierarchical representation stored in
%                             FULL_VIS (and FULL_VIS_KEY)
%         ALL_PARTIAL_KEYS -- List of vectors of lengths of repeats in the
%                             partial representations. Each vector of
%                             lengths corresponds to a partial hierarchical
%                             representation stored in ALL_PARTIAL_REPS
%         ALL_PARTIAL_WIDTHS -- List of the widths corresponding to the 
%                               partial hierarchical representations 
%                               stored in ALL_PARTIAL_REPS
%         ALL_PARTIAL_NUM_BLOCKS -- Number of repeats in each partial
%                                   representation in ALL_PARTIAL_REPS

% WARNING: DO NOT USE A VARIABLE THAT IS THE FULL BLOCK VISUALIZATION. You
%          need to use an object that ONLY has the starting time step of
%          each repeat marked like MATRIX_NO

% Set up the loop for getting all the partial representations. In this
%      loop, we exclude any representations with no hierarchical
%      information. Our first step is to remove all rows with the smallest
%      block width from consideration.

smallest_bw = min(full_vis_key);
first_row_2_check = find((full_vis_key > smallest_bw),1);
last_row_2_check = size(full_vis,1);

% Initialize the temp
all_partial_ps_rep = {};
all_partial_ps_rep_key = {};
all_partial_ps_rep_widths = [];
all_partial_ps_rep_blocks = [];

% Loop over all rows that we want to check
for i = first_row_2_check:last_row_2_check
    [partial_ps_rep_add, partial_ps_rep_key_add, ...
        partial_ps_rep_widths_add, partial_ps_rep_blocks_add] = ...
        get_mini_vis(full_vis, full_vis_key, i);
   
    all_partial_ps_rep = [all_partial_ps_rep, partial_ps_rep_add];
    all_partial_ps_rep_key = ...
        [all_partial_ps_rep_key,partial_ps_rep_key_add];
    all_partial_ps_rep_widths = ...
        [all_partial_ps_rep_widths, partial_ps_rep_widths_add];
    all_partial_ps_rep_blocks = ...
        [all_partial_ps_rep_blocks, partial_ps_rep_blocks_add];
end

% Check that the partial representations are unique and remove any copies
% of partial representations
[all_partial_reps, all_partial_keys, all_partial_widths, ...
    all_partial_num_blocks] = ...
    find_unique_partial_reps(all_partial_ps_rep, all_partial_ps_rep_key, ...
    all_partial_ps_rep_widths, all_partial_ps_rep_blocks);
end
