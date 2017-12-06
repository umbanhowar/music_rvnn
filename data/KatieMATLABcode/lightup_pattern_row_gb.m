function [pattern_row, k_lst_out, overlap_lst] = ...
    lightup_pattern_row_gb(k_mat, sn, band_width)

% Use LIGHTUP_PATTERN_ROW_GB does the following three things:
%   ONE: Turn the K_MAT into marked rows with annotation markers for the
%        start indices and 0's otherwise 
%   TWO: After removing the annotations that have overlaps, output
%        K_LST_OUT which only contains rows that have no overlaps
%   THREE: The annotations that have overlaps get removed from 
%          K_LST_OUT and gets added to OVERLAP_LST
%
% INPUT: K_MAT -- List of pairs of repeats of length BAND_WIDTH with 
%                 annotations marked. The first two columns refer to the
%                 first repeat or the pair, the second two refer to the
%                 second repeat of the pair, the fifth column refers to the
%                 length of the repeats, and the sixth column contains the
%                 annotation markers
%        SN -- Song length, which is the number of audio shingles
%        BAND_WIDTH -- The length of repeats encoded in K_MAT
%
% OUTPUT: PATTERN_ROW -- Row that marks where non-overlapping repeats
%                        occur, marking the annotation markers for the
%                        start indices and 0's otherwise
%         K_LST_OUT -- List of pairs of repeats of length BAND_WIDTH that
%                      contain no overlapping repeats with annotations
%                      marked
%         OVERLAP_LST -- List of pairs of repeats of length BAND_WIDTH that
%                        contain overlapping repeats with annotations
%                        marked

% Step 0: Initialize outputs: Start with a vector of all 0's for
%         PATTERN_ROW and assume that the row has no overlaps.
pattern_row = zeros(1,sn);
overlap_lst = [];
bw = band_width;

% Step 0a: Find the number of distinct annotations
anno_lst = k_mat(:,6);
anno_max = max(anno_lst);

% Step 1: Loop over the annotations
for a = 1:anno_max
    % Step 1a: Add 1's to PATTERN_ROW to the time steps where repeats with
    % annotation A begin
    ands = (anno_lst == a);
    start_inds = [k_mat(ands,1); k_mat(ands,3)];
    pattern_row(start_inds) = a;
    
    % Step 2: CHECK ANNOTATION BY ANNOTATION
    good_check = zeros(1,sn); % Start with row of 0's
    good_check(start_inds) = 1; % Add 1 to all time steps where repeats 
                                % with annotation A begin
    
    % Using RECONSTRUCT_FULL_BLOCK to check for overlaps
    block_check = reconstruct_full_block(good_check, bw);
    
    % If there are any overlaps, remove the bad annotations from both
    % the PATTERN_ROW and from the K_LST_OUT
    if max(block_check) > 1
        % Remove the bad annotations from PATTERN_ROW
        pattern_row(start_inds) = 0;
        
        % Remove the bad annotations from K_LST_OUT and add them to
        % OVERLAP_LST
        rm_inds = ands;
        
        temp_add = k_mat(rm_inds,:);
        overlap_lst = [overlap_lst; temp_add];
        
        k_mat(rm_inds,:) = [];
        anno_lst = k_mat(:,6);
    end
end

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

% Edit the annotations in the OVERLAP_LST so that the annotations start
% with 1 and increase one each time
if ~isempty(overlap_lst)
    overlap_lst = unique(overlap_lst, 'rows');
    [overlap_lst] = add_annotations(overlap_lst, sn);
end

end