function [lst_no_overlaps, matrix_no_overlaps, key_no_overlaps, ...
    annotations_no_overlaps, all_overlap_lst] = ...
    remove_overlaps(input_lst, song_length)

% REMOVE_OVERLAPS removes any pairs of repeat length and specific
% annotation marker where there exists at least one pair of repeats that do
% overlap in time.
%
% INPUT: INPUT_MAT -- List of pairs of repeats with annotations marked. The
%                     first two columns refer to the first repeat or the
%                     pair, the second two refer to the second repeat of
%                     the pair, the fifth column refers to the length of
%                     the repeats, and the sixth column contains the
%                     annotation markers.
%        SONG_LENGTH -- The number of audio shingles
% 
% OUTPUT: LST_NO_OVERLAPS -- List of pairs of repeats with annotations
%                            marked. All the repeats of a given length and
%                            with a specific annotation marker do not
%                            overlap in time.
%         MATRIX_NO_OVERLAPS -- Matrix representation of LST_NO_OVERLAPS
%                               with one row for each group of repeats
%         KEY_NO_OVERLAPS -- Vector containing the lengths of the repeats
%                            encoded in each row of MATRIX_NO_OVERLAPS
%         ANNOTATIONS_NO_OVERLAPS -- Vector containing the annotation
%                                    markers of the repeats encoded in each
%                                    row of MATRIX_NO_OVERLAPS
%         ALL_OVERLAP_LST -- List of pairs of repeats with annotations
%                            marked removed from INPUT_MAT. For each pair
%                            of repeat length and specific annotation
%                            marker, there exist at least one pair of
%                            repeats that do overlap in time.

L = input_lst;
sn = song_length;

bw_vec = unique(L(:,5));
bw_vec = sort(bw_vec, 'descend');

mat_NO = [];
key_NO = [];
anno_NO = [];
all_overlap_lst = [];

while ~isempty(bw_vec)
    bw = bw_vec(1);
    bsnds = find((L(:,5) == bw), 1);
    bends = find((L(:,5) > bw), 1) - 1;
    if isempty(bends)
        bends = size(L,1);
    end
    
    % Extract pairs of repeats of length BW from the list of pairs of
    % repeats with annotation markers
    bw_lst = L(bsnds:bends,:); 
                               
    L(bsnds:bends,:) = []; % remove extracted part from the annotation list 
    
    if bw > 1
        % Use LIGHTUP_PATTERN_ROW_GB to do the following three things:
        % ONE: Turn the BW_LST into marked rows with annotation markers for 
        %      the start indices and 0's otherwise 
        % TWO: After removing the annotations that have overlaps, output
        %      BW_LST_OUT which only contains rows that have no overlaps
        % THREE: The annotations that have overlaps get removed from 
        %        BW_LST_OUT and gets added to ALL_OVERLAP_LST
        [pattern_row, bw_lst_out, overlap_lst] = ...
            lightup_pattern_row_gb(bw_lst, sn, bw);
        all_overlap_lst = [all_overlap_lst; overlap_lst];
    else
        % Similar to the IF case -- 
        % Use LIGHTUP_PATTERN_ROW_BW_1 to do the following two things:
        % ONE: Turn the BW_LST into marked rows with annotation markers for 
        %      the start indices and 0's otherwise 
        % TWO: In this case, there are no overlaps. Then BW_LST_OUT is just
        %      BW_LST. Also in this case, THREE from above does not exist
        [pattern_row, bw_lst_out] = lightup_pattern_row_bw_1(bw_lst, sn);
    end
    
    if max(max(pattern_row)) > 0
        % Separate ALL annotations. In this step, we expand a row into a
        % matrix, so that there is one group of repeats per row.
        [pattern_mat, pattern_key, anno_temp_lst] = ...
            separate_all_annotations(bw_lst_out, sn, bw, pattern_row);
    else
        pattern_mat = [];
        pattern_key = [];
    end
    
    if sum(sum(pattern_mat)) > 0
        mat_NO = [mat_NO; pattern_mat];
        key_NO = [key_NO; pattern_key];
        anno_NO = [anno_NO; anno_temp_lst];
    end
    
    L = [L; bw_lst_out];
    [~, Lnds] = sort(L(:,5),'ascend');
    L = L(Lnds,:);
    
    bw_vec = unique(L(:,5));
    bw_vec = sort(bw_vec, 'descend');
    bcut = find(bw_vec < bw,1);
    
    bw_vec = bw_vec(bcut:end);
end

% Set the outputs
lst_no_overlaps = L;

if ~isempty(all_overlap_lst)
    [~,overlap_inds] = sort(all_overlap_lst(:,5), 'ascend');
    all_overlap_lst = all_overlap_lst(overlap_inds,:);
end

[key_NO, mat_inds] = sort(key_NO,'ascend');
matrix_no_overlaps = mat_NO(mat_inds,:);
key_no_overlaps = key_NO;
annotations_no_overlaps = anno_NO(mat_inds);

end