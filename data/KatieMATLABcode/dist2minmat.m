function [min_mat] = dist2minmat(distmat)

% DIST2MINMAT turns a distance matrix into a binary matrix MAX_MAT where
% MIN_MAT(i,j) == 1 if DISTMAT(i,j) < MIN(DISTMAT(i,:)) and 
% MIN_MAT(i,j) == 0 otherwise.
%
% INPUT: DISTMAT -- Distance matrix to be turned into a binary matrix
%
% OUTPUT: MIN_MAT -- Binary matrix such that MIN_MAT(i,j) == 1 if
%                    DISTMAT(i,j) < MIN(DISTMAT(i,:)) and 
%                    MIN_MAT(i,j) == 0 otherwise

myeps = 10^(-14);
D = distmat;
num_nodes = size(D,1);

min_vec = min(D,[],2);

comp_mat = repmat(min_vec,1,num_nodes);

min_mat = (abs(D-comp_mat) < myeps).*ones(num_nodes);