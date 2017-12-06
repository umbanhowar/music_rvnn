function [pattern_mat, pattern_key, anno_id_lst] = ...
    separate_all_annotations(k_mat, sn, band_width, pattern_row)

% SEPARATE_ALL_ANNOTATIONS expands PATTERN_ROW into a matrix, so that there
% is one group of repeats per row.
%
% INPUT: K_MAT -- List of pairs of repeats of length BAND_WIDTH with 
%                 annotations marked. The first two columns refer to the
%                 first repeat of the pair, the second two refer to the
%                 second repeat of the pair, the fifth column refers to the
%                 length of the repeats, and the sixth column contains the
%                 annotation markers.
%        SN -- Song length, which is the number of audio shingles
%        BAND_WIDTH -- The length of repeats encoded in K_MAT
%        PATTERN_ROW -- Row that marks where non-overlapping repeats
%                       occur, marking the annotation markers for the
%                       start indices and 0's otherwise
%
% OUTPUT: PATTERN_MAT -- Matrix representation of K_MAT with one row for 
%                        each group of repeats
%         PATTERN_KEY -- Vector containing the lengths of the repeats
%                        encoded in each row of PATTERN_MAT
%         ANNO_ID_LST -- Vector containing the annotation markers of the 
%                        repeats encoded in each row of PATTERN_MAT

bw = band_width;

% Step 0a: Find the number of distinct annotations
anno_lst = k_mat(:,6);
anno_max = max(max(anno_lst));

% Step 0b: Initialize PATTERN_MAT: Start with a matrix of all 0's that has
%          the same number of rows as there are annotations and SN columns 
pattern_mat = zeros(anno_max,sn);

% If there are two or more annotations, separate the annotations into 
% individual rows
if anno_max > 1
    for a = 1:anno_max
        ands = (anno_lst == a);
        a_starts = [k_mat(ands,1); k_mat(ands,3)];
        pattern_mat(a, a_starts) = 1; 
        pattern_key = bw*ones(anno_max,1);
    end 
else
    pattern_mat = pattern_row; 
    pattern_key = bw;
end

anno_id_lst = [1:anno_max]';

end