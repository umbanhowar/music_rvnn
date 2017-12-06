function [distAS, matAS] = ...
    cosDistMat_from_FeatureVectors(matrix_featurevecs, num_FV_per_shingle)

% COS_DISTMAT_FROM_FEATUREVECTORS creates the audio shingles for each time
% step and then takes the cosine distance between every pair of audio
% shingles. 
% 
% INPUT: MATRIX_FEATUREVECS -- Matrix of feature vectors. Each column
%                              corresponds to one time step.
%        NUM_FV_PER_SHINGLE -- Number of feature vectors per audio
%                              shingle
%
% OUTPUT: DISTAS -- Matrix of pairwise cosine distances
%          MATAS -- Matrix of audio shingles. Each column corresponds to
%                   one time step.


% No norm
M = matrix_featurevecs;
S = num_FV_per_shingle;
[n, k] = size(M);

if S == 1
    matAS = M;
else
    matAS = zeros(n*S, (k - S + 1));
    for i = 1:S
        % Use feature vectors to create an audio shingle for each time
        % step and represent these shingles as vectors, by stacking the
        % relevant feature vectors on top of each other
        matAS(((i-1)*n+1):(i*n),:) = M(:,(i:(k-S+i)));
    end
end

distASrow = pdist(matAS','cosine');
distAS = squareform(distASrow);