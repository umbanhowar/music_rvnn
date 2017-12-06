function [anno_lst_out] = get_annotation_lst(key_lst)

% GET_ANNOTATION_LST gets one annotation marker vector, given vector of
% lengths KEY_LST
%
% INPUT: KEY_LST -- Vector of lengths in ascending order
%
% OUTPUT: ANNO_LST_OUT -- Vector of one possible set of annotation markers
%                         for KEY_LST

% Initialize the temporary variable
num_rows = size(key_lst,1);
full_anno_lst = zeros(num_rows,1);

% Find the first instance of each length and give it 1 as an annotation
% marker
diff_key = key_lst - [0; key_lst(1:(num_rows - 1))];
full_anno_lst(diff_key ~= 0) = 1;

% Update CURRENT_ANNO to 2
current_anno = 2;

% Find the rows that have no annotations
anno_inds = (full_anno_lst == 0);
% Find the rows that are the same length as the row before it 
to_anno_inds = (diff_key == 0);
while sum(to_anno_inds) > 0 % Check the stopping condition - Are there rows
                            % without annotations?
    
    % Find the CURRENT_ANNO-th instance of each length
    temp_key = key_lst(to_anno_inds);
    temp_key_size = size(temp_key,1);
    diff_temp_key = temp_key - [0;temp_key(1:(temp_key_size - 1))];
    
    % Give the CURRENT_ANNO-th instance of each length CURRENT_ANNO as an
    % annotation marker
    diff_key(to_anno_inds) = diff_temp_key;
    add_inds = (((diff_key ~=0) + anno_inds) == 2);
    full_anno_lst(add_inds) = current_anno;
    
    % Update temporary variables
    anno_inds = (full_anno_lst == 0);
    to_anno_inds = (diff_key == 0);
    current_anno = current_anno + 1;
end

anno_lst_out = full_anno_lst;