function [mini_vis_lst, mini_vis_lst_key, ...
    mini_vis_lst_widths, mini_vis_lst_num_blocks] = ...
    get_mini_vis(full_matrix_no, full_vis_key, row_number)

% GET_MINI_VIS uses FULL_MATRIX_NO and FULL_VIS_KEY to get the partial
% hierarchical representations for the hierarchical representation stored
% in FULL_MATRIX_NO (and FULL_VIS_KEY) that are on top of repeats in the
% row with index ROW_NUMBER.
%
% INPUT: FULL_MATRIX_NO -- Binary matrix with 1's where hierarchical 
%                          structure begins and 0's otherwise
%        FULL_VIS_KEY -- Vector containing the lengths of the hierarchical
%                        structure encoded in FULL_MATRIX_NO
%        ROW_NUMBER -- Repeats that we are searching for partial
%                      representations on top of
% 
% OUTPUT: MINI_VIS_LST -- List of partial hierarchical representations for 
%                         the hierarchical representation stored in
%                         FULL_MATRIX_NO (and FULL_VIS_KEY) that are on top
%                         of repeats in row with index ROW_NUMBER.
%         MINI_VIS_LST_KEYS -- List of vectors of lengths of repeats in the
%                              partial representations. Each vector of
%                              lengths corresponds to a partial
%                              hierarchical representation stored in
%                              MINI_VIS_LST
%         MINI_VIS_LST_WIDTHS -- List of the widths corresponding to the 
%                                partial hierarchical representations 
%                                stored in MINI_VIS_LST
%         MINI_VIS_LST_NUM_BLOCKS -- Number of repeats in each partial
%                                    representation in MINI_VIS_LST

% WARNING: DO NOT USE A VARIABLE THAT IS THE FULL BLOCK VISUALIZATION. You
%          need to use an object that ONLY has the starts like MATRIX_NO

% Note: ROW_NUMBER is the row index for the blocks that we are searching
%       for partial representations of. Therefore, we do not need to
%       include this row in our search at all. Thus we will examine rows
%       with index number 1 to (ROW_NUMBER - 1).


% Initialize variables
bw = full_vis_key(row_number); % need BW, the width of our partial 
                               % representations to be stored in
                               % MINI_VIS_LST
timesteps = find(full_matrix_no(row_number,:));
sn = size(full_matrix_no,2);
mini_vis_mat = [];
num_mv = 0;
partial_ps_rep_lst = {};
partial_ps_rep_key_lst = {};
partial_ps_rep_width_lst = [];
partial_ps_rep_num_blocks_lst = [];

% Build MV_MASK that does not allow for overhanging blocks
mv_mask = ones((row_number - 1),bw);
for si = 1:(row_number - 1)
    mv_mask(si, (bw - full_vis_key(si) + 2 : bw)) = 0;
end

% Loop over all time steps
for i = 1:length(timesteps)
    loc = timesteps(i); % Isolate one time step
    % Isolate the relevant section of FULL_VIS
    temp_mini_vis = full_matrix_no(1:(row_number-1),loc:(loc + bw - 1));
    
    % Multiply the isolated part of FULL_VIS by MV_MASK
    temp_mini_vis = temp_mini_vis.*mv_mask;
    good_inds = (sum(temp_mini_vis,2) > 1); % Check if there are at least 
                                            % 2 blocks per row
    % In the above line of code, we are summing over the rows (not down)
    %       because repeats are encoding in rows.
    if sum(good_inds) > 0 % If any rows remain, proceed
        mini_vis_mat = [mini_vis_mat, temp_mini_vis(:)]; 
        % Lay out each representation as a column and use the resulting 
        %        matrix to determine if we have a new representation
        
        % MINI_VIS_MAT is a matrix of columns. Then MINI_VIS_MAT' creates a
        %              matrix of rows. Using UNIQUE, we check to see if the
        %              currently found representation is distinct. BUT this
        %              is only a first pass and will only find if the
        %              currently found representation is the same as a
        %              previously found one with the same annotation
        %              markers. It will NOT take into account any
        %              permutations of the rows. Therefore we need to build
        %              in such a check, which we will do in
        %              GET_ALL_MINI_VIS.
        mini_vis_mat = (unique(mini_vis_mat', 'rows'))';
        if num_mv ~= size(mini_vis_mat,2) % If the current representation 
                                          % is new, then proceed
            % Update the number of distinct representations
            num_mv = size(mini_vis_mat,2); 
            
            % Isolate the representation rows that still have repeats
            mini_vis = temp_mini_vis(good_inds,:);
            
            % Isolate the block widths that match the isolated rows
            mini_vis_key = full_vis_key(good_inds);
            
            % Add the current representation to the list of partial
            %      representations
            partial_ps_rep_lst = [partial_ps_rep_lst, mini_vis];
            
            % Add the representation key to the list of representation keys
            partial_ps_rep_key_lst = [partial_ps_rep_key_lst, mini_vis_key];
            
            % Add the representation width to the list of representation
            %      widths
            partial_ps_rep_width_lst = [partial_ps_rep_width_lst, bw];
            
            % Add the number of blocks in the partial representation to the 
            %      list of storing the number of blocks per partial
            %      representation
            partial_ps_rep_num_blocks_lst = ...
                [partial_ps_rep_num_blocks_lst, sum(sum(mini_vis))];
        end
    end
end

mini_vis_lst = partial_ps_rep_lst;
mini_vis_lst_key = partial_ps_rep_key_lst;
mini_vis_lst_widths = partial_ps_rep_width_lst;
mini_vis_lst_num_blocks = partial_ps_rep_num_blocks_lst;

end
