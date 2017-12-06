function [distAS, matAS] = ...
    DistMat_from_FeatureVectors(matrix_featurevecs, num_FV_per_shingle)

M = matrix_featurevecs;
S = num_FV_per_shingle;
[n, k] = size(M);

if S == 1
    matAS = M;
else
    matAS = zeros(n*S, (k - S +1));
    for i = 1:S
        matAS(((i-1)*n+1):(i*n),:) = M(:,(i:(k-S+i)));
    end
end

distAS = dist(matAS).^2;