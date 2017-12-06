function [pattern_block] = reconstruct_full_block(pattern_mat, pattern_key)

% RECONSTRUCT_FULL_BLOCK creates a binary matrix with a block of 1's for
% each repeat encoded in PATTERN_MAT whose length is encoded in PATTERN_KEY
%
% INPUT: PATTERN_MAT -- Binary matrix with 1's where repeats begin and 0's
%                       otherwise
%        PATTERN_KEY -- Vector containing the lengths of the repeats
%                       encoded in each row of PATTERN_MAT
%
% OUTPUT: PATTERN_BLOCK -- Binary matrix representation for PATTERN_MAT
%                          with blocks of 1's equal to the length's
%                          prescribed in PATTERN_KEY

P = pattern_mat;
K = pattern_key;
sn = size(pattern_mat,2);
sb = size(pattern_mat,1);

pattern_block = zeros(sb,sn);

for r = 1:sb
    p_row = P(r,:);
    bw = K(r);
    block_mat = zeros(bw,sn);
    for b = 1:bw
        block_mat(b,:) = [zeros(1,(b-1)), p_row(1:(end - b + 1))];
    end
    pattern_block(r,:) = sum(block_mat,1);
end

end