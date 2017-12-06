function [overlaps_yn] = check_overlaps(input_mat)

% CHECK_OVERLAPS compares every pair of rows in INPUT_MAT and checks for
% overlaps between those pairs.
%
% INPUT: INPUT_MAT -- Matrix that we are checking for overlaps
%
% OUTPUT: OVERLAPS_YN -- Binary matrix where (i,j) = 1 if row i of
%                        INPUT_MAT and row j of INPUT_MAT overlap and 0
%                        otherwise.

% Nate edit:
clearvars -except input_mat

M = input_mat;
rs = size(input_mat, 1);
ws = size(input_mat, 2);

% R_LEFT -- Every row of INPUT_MAT is repeated RS times to create a 
%           submatrix. We stack these submatrices on top of each other. 
R_left = zeros((rs^2), ws);
for i = 1:rs
    R_add = M(i,:);
    R_add_mat = repmat(R_add,rs,1);
    a = (i-1)*rs + 1;
    b = i*rs;
    R_left((a:b),:) = R_add_mat;
end

clearvars -except M rs R_left
% R_RIGHT -- Stack RS copies of INPUT_MAT on top of itself
R_right = repmat(M,rs,1);

% If INPUT_MAT is not binary, create binary temporary objects
R_left = (R_left > 0);
R_right = (R_right > 0);

R_all = sum(((R_left + R_right) == 2), 2);
R_all = (R_all > 0);

overlap_mat = reshape(R_all, rs, rs);

% If OVERLAP_MAT is symmetric, only keep the upper-triangular portion. If
% not, keep all of OVERLAP_MAT. 
check_mat = overlap_mat == overlap_mat';
if sum(sum(check_mat)) == (rs^2)
    overlap_mat = triu(overlap_mat,1);
end

overlaps_yn = overlap_mat;
    
end