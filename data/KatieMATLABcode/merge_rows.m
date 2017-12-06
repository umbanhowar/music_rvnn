function [merge_mat, merge_key] = merge_rows(input_mat, input_width)

% MERGE_ROWS merges rows of INPUT_MAT that contain at least one common
% repeat of the same piece of repeated structure.
%
% INPUT: INPUT_MAT -- Binary matrix with 1's where repeats begin and 0's
%                     otherwise
%        INPUT_WIDTH -- Length of the repeats encoded in INPUT_MAT
%
% OUTPUT: MERGE_MAT -- Binary matrix with 1's where repeats begin and 0's 
%                      otherwise created by merging rows of INPUT_MAT as
%                      appropriate
%         MERGE_KEY -- Vector containing the lengths of the repeats encoded
%                      in MERGE_MAT

% Step 0: Initialize the temporary variables
not_merge = input_mat; % Everything needs to be checked 
merge_mat = []; % Nothing has been merged yet
merge_key = [];
rs = size(not_merge, 1);
bw = input_width;

% Step 1: Check the stopping condition - Has every row been checked?
while rs > 0 
    % Step 2: Start the merge process
    % Step 2a: Choose the first row that has not been merged
    row2check = not_merge(1,:);
    r2c_mat = repmat(row2check, rs, 1); % Create a comparison matrix with 
                                        % copies of ROW2CHECK row stacked
                                        % so that R2C_MAT is the same size
                                        % as the set of rows waiting to be
                                        % merged
    
    % Step 2b: Find indices where there are any overlaps between rows that 
    %          haven't been merged yet
    merge_inds = sum(((r2c_mat + not_merge) == 2),2) > 0;
    
    % Step 2c: Union the rows that have any starting indices in common with
    %          the ROW2CHECK and remove those rows from NOT_MERGE
    union_merge = sum(not_merge(merge_inds,:), 1) > 0;
    not_merge(merge_inds,:) = [];
    
    % Step 2d: Check that the newly merged rows do not cause overlaps
    %          within the row. If there are problems, rerun COMPARE_AND_CUT
    %          on the newly merged rows.
    merge_block = reconstruct_full_block(union_merge, bw);
    if max(max(merge_block)) > 1
        [union_merge, union_merge_key] = ...
            compare_and_cut(union_merge, bw, union_merge, bw);
    else
        union_merge_key = bw;
    end
    
    % Step 2e: Add UNION_MAT to MERGE_MAT and UNION_MERGE_KEY to MERGE_KEY
    merge_mat = [merge_mat; union_merge];
    merge_key = [merge_key; union_merge_key];
    
    % Step 3: Reinitialize RS for the stopping condition
    rs = size(not_merge, 1);
end

end