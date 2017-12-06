function [max_mat] = comp2maxmat(compmat)

% COMP2MAXMAT turns a comparison matrix into a binary matrix MAX_MAT where
% MAX_MAT(i,j) == 1 if COMPMAT(i,j) >= MAX(COMPMAT(i,:)) and 
% MAX_MAT(i,j) == 0 otherwise.
%
% INPUT: COMPMAT -- Comparison matrix to be turned into a binary matrix
%
% OUTPUT: MAX_MAT -- Binary matrix such that MAX_MAT(i,j) == 1 if
%                    COMPMAT(i,j) >= MAX(COMPMAT(i,:)) and 
%                    MAX_MAT(i,j) == 0 otherwise

myeps = 10^(-14);
C = compmat;
num_nodes = size(C,1);

max_vec = max(C,[],2);
zero_inds = (max_vec < myeps);

max_vec = max_vec + zero_inds;

max_comp_mat = repmat(max_vec,1,num_nodes);
max_mat = (abs((C - max_comp_mat)) < myeps).ones(num_nodes);