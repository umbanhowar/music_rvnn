function [stretch_diag_mat] = stretch_diags(thresh_diags, band_width)

% STRETCH_DIAGS stretches entries of THRESH_DIAG into diagonals of length
% BAND_WIDTH
%
% INPUT: THRESH_DIAGS -- Binary matrix where entries equal to 1 signal the
%                        existance of a diagonal
%        BAND_WIDTH -- Length of encoded diagonals
% 
% OUTPUT: STRETCH_DIAG_MAT -- Binary matrix with diagonals of length
%                             BAND_WIDTH starting at each entry prescribed
%                             in THRESH_DIAG

b_w = band_width;
n = size(thresh_diags,1) + b_w - 1; % Size of STRETCH_DIAG_MAT

temp_song_marks_out = zeros(n);
T = thresh_diags;

[inds, jnds] = find(T);
L = length(inds);

subtemp = eye(b_w);

% Expand each entry in THRESH_DIAGS to a diagonal of length BAND_WIDTH in
% STRETCH_DIAG_MAT
for i = 1:L
    tempmat = zeros(n);
    tempmat(inds(i):(inds(i) + b_w - 1), ...
        jnds(i):(jnds(i) + b_w - 1)) = subtemp;
    temp_song_marks_out = temp_song_marks_out + tempmat;
end

% Ensure that STRETCH_DIAG_MAT is a binary matrix
stretch_diag_mat = (temp_song_marks_out > 0);

end