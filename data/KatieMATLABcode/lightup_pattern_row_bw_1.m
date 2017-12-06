function [pattern_row, k_lst_out] = lightup_pattern_row_bw_1(k_mat, sn)

% Use LIGHTUP_PATTERN_ROW_BW_1 to turn the K_MAT into marked rows with
% annotation markers for the start indices and 0's otherwise
%
% INPUT: K_MAT -- List of pairs of repeats of length 1 with annotations 
%                 marked. The first two columns refer to the first repeat
%                 of the pair, the second two refer to the second repeat of
%                 the pair, the fifth column refers to the length of the
%                 repeats, and the sixth column contains the annotation
%                 markers
%        SN -- Song length, which is the number of audio shingles
%
% OUTPUT: PATTERN_ROW -- Row that marks where non-overlapping repeats
%                        occur, marking the annotation markers for the
%                        start indices and 0's otherwise
%         K_LST_OUT -- List of pairs of repeats of length BAND_WIDTH that
%                      contain no overlapping repeats with annotations
%                      marked

% Step 0: Initialize outputs: Start with a vector of all 0's for 
%         PATTERN_ROW and assume that the row has no overlaps. 
pattern_row = zeros(1,sn);

% Step 0a: Find the number of distinct annotations
anno_lst = k_mat(:,6);
anno_max = max(anno_lst);

% Step 1: Loop over the annotations
for a = 1:anno_max
    % Step 1a: Add 1's to PATTERN_ROW to the time steps where repeats with
    % annotation A begin
    ands = (anno_lst == a);
    start_inds = [k_mat(ands,1);k_mat(ands,3)];
    pattern_row(start_inds) = a;
end

% Step 2: Check that in fact each annotation has a repeat associated to it.

inds_markers = unique(pattern_row);
if inds_markers(1) == 0
    inds_markers(1) = [];
end

if ~isempty(inds_markers)
    for na = 1:length(inds_markers)
        IM = inds_markers(na);
        if IM > na
            % Fix the annotations in PATTERN_ROW
            temp_anno = (pattern_row == IM);
            pattern_row = pattern_row - (IM*temp_anno) + (na*temp_anno);
        end
    end
end

% Edit the annotations to match the annotations in PATTERN_ROW
if ~isempty(k_mat)
    k_lst_out = unique(k_mat, 'rows');
    for na = 1:length(inds_markers)
        IM = inds_markers(na);
        if IM > na
            % Fix the annotations in K_LST_OUT
            kmat_temp_anno = (k_lst_out(:,6) == IM);
            k_lst_out(:,6) = k_lst_out(:,6) ...
                - (IM*kmat_temp_anno) + (na*kmat_temp_anno);
        end
    end
else
    k_lst_out = [];
end


end