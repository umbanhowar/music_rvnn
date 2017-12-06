function [kNNG_mat] = dist2kNNG(distmat, k)

% DIST2KNNG turns a distance matrix into a binary matrix kNNG_MAT where
% kNNG_MAT(i,j) == 1 if j is a k-nearest neighbor of i and 
% kNNG_MAT(i,j) == 0 otherwise.
%
% INPUT: DISTMAT -- Distance matrix to be turned into a binary matrix
%        K - Number of nearest neighbors
%
% OUTPUT: MIN_MAT -- Binary matrix such that kNNG_MAT(i,j) == 1 if
%                    if j is a k-nearest neighbor of i and
%                    kNNG_MAT(i,j) == 0 otherwise

myeps = 10^(-14);
D = distmat;
num_nodes = size(D,1);

k_vec = zeros(num_nodes,1);

for i = 1:num_nodes
    row_dist = sort(D(i,:),'ascend');
    k_vec(i) = row_dist(k);
end

comp_mat = repmat(k_vec,1,num_nodes);

kNNG_mat = (D <= comp_mat).*ones(num_nodes);