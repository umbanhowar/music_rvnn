%function [] = generateEssentialStructureComps(file_id_str)
    addpath('KatieMATLABcode');
    filepaths = importdata('filepaths.txt');
    
    path = filepaths{175};
    sim_mat_chroma = dlmread(strcat(path, '_sim14chroma.txt'));
    sim_mat_full = dlmread(strcat(path, '_sim14full.txt'));

    % 10% threshold on chroma mats is 0.38
    mask = (sim_mat_chroma < 0.38);
    
    full_masked = mask .* sim_mat_full;
    flip_mask = -1 * (ones(size(mask)) - mask);
    
    fmc = flip_mask + full_masked;
    % 0.2 is 10% threshold on nonnegative values in chroma mats
    structure_mat = (fmc >= 0) & (fmc < 0.2);
    
    imagesc(structure_mat);
  
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

    % Assign ONE new unique number to all the zero blocks
    PNO_color_vec(one_vec == 1) = (num_colors + 1); 
    
    song = load(strcat(path, '.mat'));
    piano_roll = song.data;
    figure
    subplot(2, 1, 1)
    imagesc(flip(piano_roll, 1));
    PNO_cv_sixteenth = repelem(PNO_color_vec, 4);
    rng = (1:length(PNO_color_vec)) .* 4;
    xticks(rng);
    xticklabels(PNO_color_vec);
    subplot(2, 1, 2)
    imagesc(PNO_cv_sixteenth)
    colormap(jet(max(PNO_color_vec) + 1));
%end