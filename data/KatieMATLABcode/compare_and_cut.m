function [union_mat, union_length] = compare_and_cut(red, RL, blue, BL)

% COMPARE_AND_CUT compares two rows of repeats labeled RED and BLUE, and
% determines if there are any overlaps in time between them. If there is,
% then we cut the repeats in RED and BLUE into up to 3 pieces. 
%
% INPUT: RED -- Binary row vector encoding a set of repeats with 1's where
%               each repeat starts and 0's otherwise
%        RL -- Length of repeats encoded in RED
%        BLUE -- Binary row vector encoding a set of repeats with 1's where
%                each repeat starts and 0's otherwise
%        BL -- Length of repeats encoded in BLUE
%
% OUTPUT: UNION_MAT -- Binary matrix of up to three rows encoding
%                      non-overlapping repeats cut from RED and BLUE
%         UNION_LENGTH -- Vector containing the lengths of the repeats 
%                         encoded in UNION_MAT

sn = size(red,2); assert(sn == size(blue,2));
start_red = find(red);
start_blue = find(blue);

% Determine if the rows have any intersections
red_block = reconstruct_full_block(red, RL);
blue_block = reconstruct_full_block(blue, BL);
red_block = (red_block > 0);
blue_block = (blue_block > 0);
purple_block = ((red_block + blue_block) == 2);

% If there is any intersection between the rows, then start comparing one
% repeat in RED to one repeat in BLUE
if sum(sum(purple_block)) > 0
    % Find number of blocks in red and in blue
    LSR = length(start_red);
    LSB = length(start_blue);
    
    % Build the pairs of starting indices to search, where each pair
    % contains a starting index in RED and a starting index in BLUE
    red_inds = repmat(start_red',LSB,1);
    blue_inds = repmat(start_blue,LSR,1);
    compare_inds = [blue_inds(:),red_inds];
    
    % Initialize the output variables UNION_MAT and UNION_LENGTH
    union_mat = [];
    union_length = [];
    
    % Loop over all pairs of starting indices
    for ci = 1:(LSR*LSB);
        
        % Isolate one repeat in RED and one repeat in BLUE
        ri = compare_inds(ci,2);
        bi = compare_inds(ci,1);
        red_ri = (ri:1:(ri+RL-1));
        blue_bi = (bi:1:(bi+BL-1));
        
        % Determine if the blocks intersect and call the intersection
        % PURPLE
        purple = intersect(red_ri,blue_bi);
        
        % If the blocks intersect (i.e. if PURPLE is not empty), then
        % proceed
        if ~isempty(purple)
            
            % Remove PURPLE from RED_RI, call it RED_MINUS_PURPLE
            red_minus_purple = setdiff(red_ri,purple);
            
            % If RED_MINUS_PURPLE is not empty, then see if there are one
            % or two parts in RED_MINUS_PURPLE. Then cut PURPLE out of ALL
            % of the repeats in RED. If there are two parts left in
            % RED_MINUS_PURPLE, then the new variable NEW_RED, which holds
            % the part(s) of RED_MINUS_PURPLE, should have two rows with
            % 1's for the starting indices of the resulting pieces and 0's
            % elsewhere. Also RED_LENGTH_VEC will have the length(s) of the
            % parts in NEW_RED.
            if ~isempty(red_minus_purple)
                [red_start_mat, red_length_vec] = ...
                    num_of_parts(red_minus_purple, ri, start_red);
                [new_red] = inds_to_rows(red_start_mat,sn);
            else
                % If RED_MINUS_PURPLE is empty, then set NEW_RED and
                % RED_LENGTH_VEC to empty
                new_red = [];
                red_length_vec = [];
            end
            
            % Noting that PURPLE is only one part and in both RED_RI and
            % BLUE_BI, then we need to find where the purple starting
            % indices are in all the RED_RI
            [purple_in_red_mat, purple_length] = ...
                num_of_parts(purple, ri, start_red);
            
            % If BLUE_MINUS_PURPLE is not empty, then see if there are one
            % or two parts in BLUE_MINUS_PURPLE. Then cut PURPLE out of ALL
            % of the repeats in BLUE. If there are two parts left in
            % BLUE_MINUS_PURPLE, then the new variable NEW_BLUE, which
            % holds the part(s) of BLUE_MINUS_PURPLE, should have two rows
            % with 1's for the starting indices of the resulting pieces and
            % 0's elsewhere. Also BLUE_LENGTH_VEC will have the length(s)
            % of the parts in NEW_BLUE.
            blue_minus_purple = setdiff(blue_bi,purple);
            if ~isempty(blue_minus_purple)
                [blue_start_mat, blue_length_vec] = ...
                    num_of_parts(blue_minus_purple, bi, start_blue);
                [new_blue] = inds_to_rows(blue_start_mat, sn);
            else
                % If BLUE_MINUS_PURPLE is empty, then set NEW_BLUE and
                % BLUE_LENGTH_VEC to empty
                new_blue = [];
                blue_length_vec = [];
            end
            
            % Recalling that PURPLE is only one part and in both RED_RI and
            % BLUE_BI, then we need to find where the purple starting
            % indices are in all the BLUE_RI
            [purple_in_blue_mat] = ...
                num_of_parts(purple, bi, start_blue);
            
            % Union PURPLE_IN_RED_MAT and PURPLE_IN_BLUE_MAT to get
            % PURPLE_START, which stores all the purple indices
            purple_start = ...
                union(purple_in_red_mat, purple_in_blue_mat);
            
            % Use PURPLE_START to get NEW_PURPLE with 1's where the repeats
            % in the purple rows start and 0 otherwise. 
            [new_purple] = inds_to_rows(purple_start, sn);
            
            if ~isempty(new_red) || ~isempty(new_blue)
                % Form the outputs
                union_mat = [new_red; new_blue; new_purple];
                union_length = [red_length_vec; ...
                    blue_length_vec; purple_length];
                U = union_mat;
                L = union_length;
                [union_mat, union_length] = merge_based_on_length(U, L, L);
                break
            elseif isempty(new_red) && isempty(new_blue)
                new_purple_block = ...
                    reconstruct_full_block(new_purple, purple_length);
                if max(max(new_purple_block)) < 2
                    union_mat = new_purple;
                    union_length = purple_length;
                    break
                end
            end
            
        end
    end
end

% Check that there are no overlaps in each row of UNION_MAT
union_mat_add = [];
union_mat_add_length = [];
union_mat_rminds = [];

% Isolate one row at a time, call it UNION_ROW
for ur = 1:size(union_mat,1)
    union_row = union_mat(ur,:);
    union_row_width = union_length(ur);
    union_row_block = reconstruct_full_block(union_row, union_row_width);
    
    % If there are at least one overlap, then compare and cut that row
    % until there are no overlaps
    if sum(union_row_block > 1) > 0
        union_mat_rminds = [union_mat_rminds; ur];
        
        [union_row_new, union_row_new_length] = ...
            compare_and_cut(union_row, union_row_width,...
            union_row, union_row_width);
        
        % Add UNION_ROW_NEW and UNION_ROW_NEW_LENGTH to UNION_MAT_ADD and
        % UNION_MAT_ADD_LENGTH, respectively
        union_mat_add = [union_mat_add;union_row_new];
        union_mat_add_length = ...
            [union_mat_add_length; union_row_new_length];
        
    end
end

% Remove the old rows from UNION_MAT (as well as the old lengths from
% UNION_LENGTH)
union_mat(union_mat_rminds,:) = [];
union_length(union_mat_rminds) = [];

% Add UNION_ROW_NEW and UNION_ROW_NEW_LENGTH to UNION_MAT and
% UNION_LENGTH, respectively, such that UNION_MAT is in order by
% lengths in UNION_LENGTH
union_mat = [union_mat;union_mat_add];
union_length = [union_length; union_mat_add_length];

[union_length, UM_inds] = sort(union_length,'ascend');
union_mat = union_mat(UM_inds,:);

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Additional sub-functions needed for above function:                     %
%                                                                         %
% function NUM_OF_PARTS determines the number of blocks of consecutive    %
% time steps in a given list of time steps                                %
%          INPUT: INPUT_VEC -- One or two parts to replicate              %
%                 INPUT_START -- Starting index for part to be replicated %
%                 INPUT_ALL_STARTS -- Starting indices for replication    %
%          OUTPUT: START_MAT -- Matrix of one or two rows, containing the %
%                               starting indices                          %
%                  LENGTH_VEC -- Column vector of the lengths             %
%                                                                         %
% function INDS_TO_ROWS converts a list of indices to row(s) with 1's     %
% where an index occurs and 0's otherwise                                 %
%          INPUT: INPUT_INDS_MAT -- Matrix of one or two rows, containing %
%                                   the starting indices                  %
%                 ROW_LENGTH -- length of the rows                        %
%          OUTPUT: MAT_OUT -- Matrix of one or two rows, with 1's where   %
%                             the starting indices and 0's otherwise      %
%                                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [start_mat, length_vec] = ...
    num_of_parts(input_vec, input_start, input_all_starts)

    diff_vec = input_vec(2:end) - input_vec(1:end-1);
    diff_vec = [1,diff_vec];
    break_mark = find(diff_vec > 1);

    if isempty(break_mark)
        start_vec = input_vec(1);
        end_vec = input_vec(end);

        add_vec = start_vec - input_start;
        start_mat = input_all_starts + add_vec;
    else
        start_vec = zeros(2,1);
        end_vec = zeros(2,1);

        start_vec(1) = input_vec(1);
        end_vec(1) = input_vec(break_mark - 1);

        start_vec(2) = input_vec(break_mark);
        end_vec(2) = input_vec(end);

        add_vec = start_vec - input_start;
        start_mat = [(input_all_starts + add_vec(1)); ...
            (input_all_starts + add_vec(2))];
    end

    length_vec = end_vec - start_vec + 1;

end

function [mat_out] = inds_to_rows(input_inds_mat, row_length)

    inds_mat = input_inds_mat;
    r = row_length;

    mat_rows = size(inds_mat,1);
    new_mat = zeros(mat_rows,r);

    for j = 1:mat_rows
        jnds = inds_mat(j,:);
        new_mat(j,jnds) = 1;
    end
    mat_out = new_mat;

end