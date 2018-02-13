function [] = generateEssentialStructureComps(file_id_str)
    write = true;
    filetype = '6_4chroma';
    thresh = 0.1;

    addpath('KatieMATLABcode');
    filepaths = importdata('filepaths.txt');
    
    % 908 interesting
    path = filepaths{str2num(file_id_str)};
    sim_mat = dlmread(strcat(path, '_sim', filetype, '.txt'));

    structure_mat = (sim_mat < thresh);
    
    % If no essential structure components, just write right away
    if structure_mat == eye(size(sim_mat, 1))
        if write
            write_ESC([size(sim_mat, 1) 1], path, filetype, num2str(thresh));
        end
        return
    end
    
    if ~write
        figure
        imagesc(sim_mat);
        figure
        imagesc(structure_mat);
    end
    
    if ~write
        figure
        song = load(strcat(path, '.mat'));
        piano_roll = song.data;
        imagesc(flip(piano_roll, 1));
    end
    
    sn = size(structure_mat, 1);
    TDDM = structure_mat .* ones(sn);
    
    clear sim_mat_chroma sim_mat_full mask full_masked flip_mask fmc structure_mat
    
    % Extract all diagonals in TDDM, saving the pairs that the diagonals
    % represent
    all_lst = lightup_lst_with_thresh_bw(TDDM, (1:sn), 0);
    
    % Find smaller repeats that are contained in larger repeats in ALL_LST
    lst_gb = find_complete_list(all_lst, sn);
    
    % Remove groups of repeats that contain at least two repeats that
    % overlap in time
    [~, matrix_no, key_no, ~, ~] = remove_overlaps(lst_gb, sn);
    
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
    
    
    starts_marked = PNO + PNO_block;
    
    if ~write
        figure
        imagesc(starts_marked);
    end    
    
    % Convert the ESC representation with their starts marked to one with
    % their ends marked.
    for ii=1:size(starts_marked, 1)
        row = starts_marked(ii, :);
        for jj=2:size(starts_marked, 2)
            % 2 2 - Both should be marked as ends (do nothing)
            % 2 1 - First is a start, decrement
            % 1 2 - First is an end, increment
            % 1 1 - Last is an end if at end of song
            % 2 0 - First is an end, do nothing
            % 0 2 - Do nothing
            % 1 0 - First is an end, increment
            % 0 1 - Doesn't happen
            % 0 0 - Do nothing
            first = row(jj-1);
            last = row(jj);
            if first == 2 && last == 1
                row(jj-1) = 1;
            elseif first == 1 && last == 2
                row(jj-1) = 2;
            elseif first == 1 && last == 0
                row(jj-1) = 2;
            end
            if jj == size(starts_marked, 2) && row(jj) ~= 0
                row(jj) = 2;
            end
        end
        starts_marked(ii, :) = row;
    end
    ends_marked = starts_marked;
    
    % Assign a unique ID to each nonstructural components and mark their end
    % times.
    in_block = false;
    new_row = zeros(1, size(ends_marked, 2));
    for ii=1:size(ends_marked, 2)
        % If you get a column of all zeros
        if ~any(ends_marked(:, ii))
            if ~in_block
                in_block = true;
            end
            % If we're at the last position, mark the end.
            if ii == size(ends_marked, 2)
                new_row(ii) = 2;
                ends_marked = [ends_marked; new_row];
            else
                % Otherwise mark that we're in the component.
                new_row(ii) = 1;
            end
        else
            if in_block
                in_block = false;
                new_row(ii-1) = 2;
                ends_marked = [ends_marked; new_row];
                new_row = zeros(1, size(ends_marked, 2));
            end
        end
    end
    
    if ~write
        figure
        imagesc(ends_marked);
        figure
        imagesc(sum(ends_marked, 1));
    end
    
    % Convert to correct representation format and write to file.
    ESC = [];
    for ii=1:size(ends_marked, 2)
        for jj=1:size(ends_marked, 1)
            if ends_marked(jj, ii) == 2
                ESC = [ESC; [ii jj]];
            end
        end
    end
    
    if write
        write_ESC(ESC, path, filetype, num2str(thresh));
    end
%end