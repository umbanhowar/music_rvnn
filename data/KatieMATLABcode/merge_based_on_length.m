function [out_mat, out_length_vec] = ...
    merge_based_on_length(full_mat, full_bw, target_bw)

% MERGE_BASED_ON_LENGTH merges rows of FULL_MAT that contain repeats that
% are the same length and are repeats of the same piece of structure
%
% INPUT: FULL_MAT -- Binary matrix with 1's where repeats begin and 0's
%                    otherwise
%        FULL_BW -- Length of the repeats encoded in FULL_MAT
%        TARGET_BW -- Lengths of repeats that we seek to merge
%
% OUTPUT: OUT_MAT -- Binary matrix with 1's where repeats begin and 0's
%                    otherwise with rows of FULL_MAT merged if appropriate
%         OUT_LENGTH_VEC -- Length of the repeats encoded in OUT_MAT

[temp_bw, bnds] = sort(full_bw, 'ascend');
temp_mat = full_mat(bnds,:);

% Find the list of unique lengths that you would like to search
target_bw = unique(target_bw);
T = size(target_bw,1);

% Loop over all unique repeat search lengths
for i = 1:T
    test_bw = target_bw(i);
    inds = (temp_bw == test_bw);
    if sum(sum(inds)) > 1
        % Isolate rows that correspond to TEST_BW and merge them
        toBmerged = temp_mat(inds, :);
        merged_mat = merge_rows(toBmerged, test_bw);
        
        bw_add_size = size(merged_mat,1);
        bw_add = test_bw*ones(bw_add_size,1);
        
        temp_mat(inds,:) = [];
        temp_bw(inds,:) = [];
        
        temp_mat = [temp_mat; merged_mat];
        temp_bw = [temp_bw; bw_add];
        
        [temp_bw, bnds] = sort(temp_bw, 'ascend');
        temp_mat = temp_mat(bnds,:);
    end
end

out_mat = temp_mat;
out_length_vec = temp_bw;
end