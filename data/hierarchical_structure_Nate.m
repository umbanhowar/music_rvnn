function [full_visualization, full_key, full_matrix_no, full_anno_lst] ...
    = hierarchical_structure(matrix_no, key_no, sn)

% HIERARCHICAL_STRUCTURE distills the repeats encoded in MATRIX_NO (and
% KEY_NO) to the essential structure components and then builds the
% hierarchical representation
%
% INPUT: MATRIX_NO -- Binary matrix with 1's where repeats begin and 0's
%                     otherwise
%        KEY_NO -- Vector containing the lengths of the repeats encoded in
%                  MATRIX_NO
%        SN -- Song length, which is the number of audio shingles
%
% OUTPUT: FULL_VISUALIZATION -- Binary matrix representation for
%                               FULL_MATRIX_NO with blocks of 1's equal to
%                               the length's prescribed in FULL_KEY
%         FULL_KEY -- Vector containing the lengths of the hierarchical
%                     structure encoded in FULL_MATRIX_NO
%         FULL_MATRIX_NO -- Binary matrix with 1's where hierarchical 
%                           structure begins and 0's otherwise
%         FULL_ANNO_LST -- Vector containing the annotation markers of the 
%                          hierarchical structure encoded in each row of
%                          FULL_MATRIX_NO

% Distill repeats encoded in MATRIX_NO (and KEY_NO) to the essential
% structure components, the set of repeated structure so that no time step
% is contained in more than one piece of structure. We call the resulting
% matrix PNO (for pattern no overlaps) and the associated key PNO_KEY.
[PNO, PNO_key] = breakup_overlaps_by_intersect(matrix_no, key_no, 0);

% Using PNO and PNO_KEY, we build a vector that tells us the order of the
% repeats of the essential structure components.

% Get the block representation for PNO, called PNO_BLOCK
[PNO_block] = reconstruct_full_block(PNO, PNO_key);

% Assign a unique number for each row in PNO. We refer these unique numbers
% COLORS. 
num_colors = size(PNO,1);
num_timesteps = size(PNO,2);
color_mat = repmat([1:num_colors]', 1, num_timesteps);

% For each time step in row i that equals 1, change the value at that time
% step to i
PNO_color = color_mat.*PNO;
PNO_color_vec = sum(PNO_color,1);

% Find where repeats exist in time, paying special attention to the starts
% and ends of each repeat of an essential structure component
PNO_block_vec = sum(PNO_block,1) > 0;
one_vec = (PNO_block_vec(1:sn-1) - PNO_block_vec(2:sn)); 
    % ONE_VEC -- If ONE_VEC(i) = 1, means that there is a block that ends
    %            at t_i and that there is no block that starts at t_{i+1}

% Find all the blocks of consecutive time steps that are not contained in
% any of the essential structure components. We call these blocks zero
% blocks. 
% Shift PNO_BLOCK_VEC so that the zero blocks are marked at the correct
% time steps with 1's
if PNO_block_vec(1) == 0 % There is no block at time step 1
    one_vec = [1,one_vec];
elseif PNO_block_vec(1) == 1 % There is a block at time step 1
    one_vec = [0,one_vec];
end

% Assign ONE new unique number to all the zero blocks
PNO_color_vec(one_vec == 1) = (num_colors + 1); 

% We are only concerned with the order that repeats of the essential
% structure components occur in. So we create a vector that only contains
% the starting indices for each repeat of the essential structure
% components.

% We isolate the starting index of each repeat of the essential structure
% components and save a binary vector with 1 at a time step if a repeat of
% any essential structure component occurs there
non_zero_inds = (PNO_color_vec > 0);
num_NZI = sum(non_zero_inds); % want the number of repeats, not a vector. 
PNO_color_inds_only = PNO_color_vec(non_zero_inds);

% For indices that signals the start of a zero block, turn those indices
% back to 0
zero_inds_short = (PNO_color_inds_only == (num_colors + 1));
PNO_color_inds_only(zero_inds_short) = 0;

% Create a binary matrix SYMM_PNO_INDS_ONLY such that the (i,j) entry is 1
% if the following three conditions are true: 
%     1) a repeat of an essential structure component is the i-th thing in
%        the ordering
%     2) a repeat of an essential structure component is the j-th thing in 
%        the ordering 
%     3) the repeat occurring in the i-th place of the ordering and the one
%        occuring in the j-th place of the ordering are repeats of the same
%        essential structure component. 
% If any of the above conditions are not true, then the (i,j) entry of
% SYMM_PNO_INDS_ONLY is 0.

% Turn our pattern row into a square matrix by stacking that row the
% number of times equal to the columns in that row
PNO_IO_mat = repmat(PNO_color_inds_only, num_NZI, 1);

% Check that the i-th ordering and the j-th ordering are both repeats of
% essential structure components (i.e. conditions 1 and 2 above)
PNO_IO_mask = (((PNO_IO_mat > 0) + (PNO_IO_mat' > 0)) == 2);

% Check that the repeat occurring in the i-th place of the ordering and the
% one occuring in the j-th place of the ordering are repeats of the same
% essential structure component (i.e. condition 3 above) combined with the
% check above. The result is the desired binary matrix. 
symm_PNO_inds_only = (PNO_IO_mat == PNO_IO_mat').*PNO_IO_mask;

% Extract all the diagonals in SYMM_PNO_INDS_ONLY and get pairs of repeated
% sublists in the order that repeats of essential structure components.
% These pairs of repeated sublists are the basis of our hierarchical
% representation.
[NZI_lst] = ...
    lightup_lst_with_thresh_bw_no_remove(symm_PNO_inds_only, [1:num_NZI]); 

% Remove any pairs of repeats that are two copies of the same repeat (i.e.
% a pair (A,B) where A == B)
remove_inds = (NZI_lst(:,1) == NZI_lst(:,3));
NZI_lst(remove_inds,:) = [];

% Add the annotation markers to the pairs in NZI_LST
[NZI_lst_anno] = find_complete_list_anno_only(NZI_lst, num_NZI);

% Remove any groups of the repeats that contain at least two overlapping
% repeats and transition from a list of pairs of repeats to a matrix
% representation (NZI_MATRIX_NO) with two vectors (NZI_KEY_NO and
% NZI_ANNO_NO) acting as the key for the matrix representation. This
% triplet is the abbreviated version of our hierarchical representation.
[~, NZI_matrix_no, NZI_key_no] = remove_overlaps(NZI_lst_anno, num_NZI);

% Get the block representation for NZI_MATRIX_NO (with NZI_KEY_NO)
[NZI_pattern_block] = reconstruct_full_block(NZI_matrix_no, NZI_key_no);

% Now we begin the process of expanding the abbreviated version of our
% hierarchical representation (NZI_MATRIX_NO, NZI_KEY_NO and NZI_ANNO_NO)
% to the full hierarchical representation

% Find where all blocks start and end
[nzi_rows] = size(NZI_pattern_block,1);
pattern_starts = find(non_zero_inds);
pattern_ends = [(pattern_starts(2:end) - 1), sn];
pattern_lengths = [pattern_ends - pattern_starts + 1];

% Create the FULL_VISUALIZATION and FULL_MATRIX_NO by copying the i-th
% column of the NZI_PATTERN_BLOCK and NZI_MATRIX_NO, respectively, the
% length of the i-th repeat of essential structure in our ordering,
% inserting these copies where the the i-th repeat of essential structure
% in our ordering begins
full_visualization = zeros(nzi_rows, sn);
full_matrix_no = zeros(nzi_rows, sn);

for i = 1:num_NZI
    full_visualization(:,[pattern_starts(i):pattern_ends(i)]) = ...
        repmat(NZI_pattern_block(:,i),1,pattern_lengths(i));
    full_matrix_no(:,pattern_starts(i)) = NZI_matrix_no(:,i);
end

% Get FULL_KEY, the matching bandwidth key for FULL_MATRIX_NO
full_key = zeros(nzi_rows,1);
find_key_mat = full_visualization + full_matrix_no;
for i = 1:nzi_rows
    one_start = find(find_key_mat(i,:) == 2,1);
    temp_row = find_key_mat(i,:);
    temp_row(1:one_start) = 1;
    find_zero = find(temp_row == 0,1);
    if isempty(find_zero)
        find_zero = sn + 1; % why SN + 1? Because we do not want to change 
    end                     %             the block size. In this loop, 0 
                            %             means no block at that time step,
                            %             1 means a block covers that time
                            %             step, and 2 means that a block
                            %             starts at that time step.
    
    find_two = find(temp_row == 2,1);
    if isempty(find_two)
        find_two = sn + 1;
    end
    one_end = min(find_zero,find_two);
    full_key(i) = one_end - one_start;
end

[full_key, full_key_inds] = sort(full_key,'ascend');
full_visualization = full_visualization(full_key_inds,:);
full_matrix_no = full_matrix_no(full_key_inds,:);

% Remove rows of our hierarchical representation that contain only one
% repeat
inds_remove = (sum(full_matrix_no,2) <= 1);
full_key(inds_remove) = [];
full_matrix_no(inds_remove,:) = [];
full_visualization(inds_remove,:) = [];

% Get FULL_ANNO_LST, the matching annotation key for FULL_MATRIX_NO
[full_anno_lst] = get_annotation_lst(full_key);
