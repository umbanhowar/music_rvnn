function [distAS, matAS] = ...
    cosDistMat_from_FeatureVectors_Nate(matrix_featurevecs, ...
    num_FV_per_shingle, FV_hop)

% COS_DISTMAT_FROM_FEATUREVECTORS_NATE creates the audio shingles for each 
% time step and then takes the cosine distance between every pair of audio
% shingles. 
% 
% INPUT: MATRIX_FEATUREVECS -- Matrix of feature vectors. Each column
%                              corresponds to one time step.
%        NUM_FV_PER_SHINGLE -- Number of feature vectors per audio
%                              shingle
%                    FV_HOP -- The number of FV that are skipped over when
%                              making MATAS
%
% OUTPUT: DISTAS -- Matrix of pairwise cosine distances
%          MATAS -- Matrix of audio shingles. Each column corresponds to
%                   one time step.


% No norm
M = matrix_featurevecs;
S = num_FV_per_shingle;
[n, k] = size(M);
h = FV_hop;

% Create FV_MAT_HOPPED that stacks all the feature vectors within the hop
% into one long feature vector. 
fn = n*h;
fk = ceil(k/h);
FV_mat_hopped = zeros(n*h,ceil(k/h));

% Check that the number of feature vectors divides nicely. Otherwise, pad
% the end of the matrix with zeros. 
if ceil(k/h) ~= k/h
    pad = h*(ceil(k/h) - k/h);
    M = [M,zeros(n,pad)];
end

for f = 1:ceil(k/h)
    M_slice = M(:,((f-1)*h+1):(f*h));
    FV_mat_hopped(:,f) = M_slice(:);
end



if S == 1
    matAS = FV_mat_hopped;
else
    matAS = zeros(fn*S, (fk - S + 1));
    for i = 1:S
        % Use feature vectors to create an audio shingle for each time
        % step and represent these shingles as vectors, by stacking the
        % relevant feature vectors on top of each other
        matAS(((i-1)*fn+1):(i*fn),:) = FV_mat_hopped(:,(i:(fk-S+i)));
    end
end

distASrow = pdist(matAS','cosine');
distAS = squareform(distASrow);