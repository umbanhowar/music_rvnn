function [thresh] = find_thresh(thresh_flag, max_val)

% FIND_THRESH finds a threshold based on what we would use to determine if
% two audio shingles are similar: angle between the audio shingles,
% proportion of the length of the orthogonal projection compared to the
% norm of an audio shingle, or a threshold set without consideration of the
% previously stated options.
%
% INPUT: THRESH_FLAG -- denotes which kind of value we are basing our
%                       threshold on: 'p' for proportion of the length of
%                       the orthogonal projection compared to the norm of
%                       an audio shingle, 'a' for angle between the audio
%                       shingles, and 't' for a threshold set without
%                       direct consideration of either 'a' or 'p'
%        MAX_VAL -- denotes the largest accepted value.
%
% OUTPUT: THRESH -- Threshold (to be used in later scripts)

% For two vectors A and B, the dist(A,B) = 1 - cos(theta), where theta is
%     the angle between A and B.

if strcmp(thresh_flag,'p')
    thresh = 1 - 1/sqrt((max_val^2)+1);
elseif strcmp(thresh_flag,'a')
    thresh = 1 - cos(max_val);
elseif strcmp(thresh_flag,'t')
    thresh = max_val;
else
    fprintf('ERROR');
end