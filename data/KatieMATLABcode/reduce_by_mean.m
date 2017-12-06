function [mean_feature_vectors] = ...
    reduce_by_mean(feature_vectors, num_feature_vectors)

% REDUCE_BY_MEAN reduces the number of columns in FEATURE_VECTORS by taking
% the average of NUM_FEATURE_VECTORS number of concatenated feature vectors
% in FEATURE_VECTORS. The output MEAN_FEATURE_VECTORS will have the same
% number of rows as FEATURE_VECTORS but will have
% ROUND(SIZE(FEATURE_VECTORS,2)/NUM_FEATURE_VECTORS) number of columns.
%
% INPUT: FEATURE_VECTORS -- Matrix with feature vectors as columns
%        NUM_FEATURE_VECTORS -- Number of feature vectors to be averaged to
%                               create one row in MEAN_FEATURE_VECTORS
%
% OUTPUT: MEAN_FEATURE_VECTORS -- Matrix with averaged feature vectors as
%                                 columns

F = feature_vectors;
Fc = size(feature_vectors,2);
Fr = size(feature_vectors,1);

nfv = num_feature_vectors;
Mc = round(Fc/nfv);
M = zeros(Fr,Mc);

for i = 1:(Mc - 1)
    Fslice = F(:,((i-1)*nfv + 1):(i*nfv));
    M(:,i) = mean(Fslice,2);
end

Fslice = F(:,((Mc-1)*nfv + 1):Fc);
M(:,Mc) = mean(Fslice,2);

mean_feature_vectors = M;

