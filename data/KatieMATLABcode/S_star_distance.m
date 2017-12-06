function [dist_LcR] = S_star_distance(L,R)

% S_STAR_DISTANCE computes the distance between two matrices (with the same
% number of columns) that encode the same size repetitions. We put the
% representation with the larger number of rows on the left and call it L.
%
% INPUT: L -- One matrix encoding representations
%        R -- Second matrix encoding representations. R has the same number
%             of columns as L, but has the same number or less rows than L.
% 
% OUTPUT: DIST_LCR -- The distance between L and R

% Note: S_STAR_DISTANCE is very similar to S_STAR_DISTANCE_REPS

numLrows = size(L,1); % We will fix the order of the rows of the matrix L. 
                      % By Lemma*, we know that there exists an permutation
                      % of R that will minimize the distance between L and 
                      % R. The matrix L also has at least as many rows as
                      % the matrix R. 
numRrows = size(R,1); 
rep_width = size(L,2); assert(size(L,2) == size(R,2), ...
                            'ERROR: Representations are different sizes.');
                        
% Begin by removing the rows in L that have an exact match in R. 
%       Simultaneously remove the rows in R that have an exact match in L.
Rcompmat = zeros(numLrows*numRrows,rep_width);
for i = 1:numRrows
    Rcompmat(((i-1)*numLrows + 1):(i*numLrows),:) ...
        = repmat(R(i,:),numLrows,1);
end
Lcompmat = repmat(L,numRrows,1);
% This is the distance between the rows of L and R
dist_LcR_rows = reshape(sum(abs(Lcompmat - Rcompmat),2),numLrows,numRrows);

[L_exact, R_exact] = find(dist_LcR_rows == 0);

L_ne_rows = setdiff([1:numLrows],L_exact);
R_ne_rows = setdiff([1:numRrows],R_exact);

num_L_ne = length(L_ne_rows);
num_R_ne = length(R_ne_rows);
    
% Case 1 -- [L] == [R] (The matrices L and R are identical)
if (num_L_ne == 0) && (num_R_ne == 0)
    dist_LcR = 0;
    
% Case 2 -- [L] > [R] (The matrix R is contained in L)
elseif (num_L_ne > 0) && (num_R_ne == 0) 
    dist_LcR = sum(sum(L(L_ne_rows,:)));
    
% Case 3 -- Matrices L and R are not contained in each other
elseif (num_R_ne > 0)
    % Isolate information for the representations that do not have exact
    %       matches that are needed for the below comparison.
    L_num_blocks_ne = sum(L(L_ne_rows,:),2);
    
    % We also need to isolate the distances in DIST_LCR_ROWS that are
    %       associated with the above isolated rows of L and R. First
    %       remove the rows that are associated with rows in L
    %       that have exact matches in R. Next remove the columns that are 
    %       associated with rows in R that have exact matches in L.
    partial_dist_mat_LcR_rows = dist_LcR_rows;
    partial_dist_mat_LcR_rows(L_exact,:) = [];
    partial_dist_mat_LcR_rows(:,R_exact) = [];

% Finally, find the distance between L and R
    if (num_R_ne <= 5)
        dist_LcR = S_star_partial_H_dist(L_num_blocks_ne, ...
                    partial_dist_mat_LcR_rows);
    elseif (num_R_ne > 5)
        dist_LcR = S_star_distance_H_split(L_num_blocks_ne, ...
                    partial_dist_mat_LcR_rows);
    end
end