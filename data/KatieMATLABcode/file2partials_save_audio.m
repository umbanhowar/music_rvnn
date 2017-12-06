function file2partials_save_audio(inname, outname, num_FV_per_shingle, thresh)

% FILE2PARTIALS_SAVE_AUDIO processes a specific audio file from the
% original feature vectors to the partial representations and saves the
% partial representations in OUTNAME.
%
% INPUT: INNAME -- Name of file to be processed
%        OUTNAME -- Name of file where partials will be saved
%        NUM_FV_PER_SHINGLE -- Number of feature vectors per shingle
%        THRESH -- Maximum threshold value
%
% OUTPUT: None. File is saved.

% Load file of feature vectors
infile = load(inname);
Fmat = infile.f_CP;

% Reduce the size of FMAT_NO_BLOCKS such that each column of FMAT_MEAN
% corresponds to 0.5 seconds of feature vectors
Fmat_mean = reduce_by_mean(Fmat,5);

% Get pairwise distance matrix using ``cosine distance''
[distAS] = cosDistMat_from_FeatureVectors(Fmat_mean, num_FV_per_shingle);

% Get the thresholded distance matrix using THRESH
sn = size(distAS,1);
TDDM = (distAS <= thresh).*ones(sn); % Multiply by ONES(SN) to make the 
                                     % resulting matrix of the DOUBLE class 

% Extract all diagonals in TDDM, saving the pairs that the diagonals
% represent
all_lst = lightup_lst_with_thresh_bw(TDDM, (1:sn), 0);

% Find smaller repeats that are contained in larger repeats in ALL_LST
lst_gb = find_complete_list(all_lst, sn);


if ~isempty(lst_gb)
    % Remove groups of repeats that contain at least two repeats that
    % overlap in time
    [~, matrix_no, key_no, ~, ~] = remove_overlaps(lst_gb, sn);
    
    % Distill the repeats encoded in MATRIX_NO and KEY_NO into the
    % essential structure components and use them to build the hierarchical
    % representation
    [~, full_key, full_matrix_no, ~] ...
        = hierarchical_structure(matrix_no, key_no, sn);
    
    % Use FULL_KEY and FULL_MATRIX_NO to get the partial hierarchical
    % representations
    [partial_reps, partial_keys, partial_widths, partial_num_blocks] ...
        = get_all_mini_vis(full_matrix_no, full_key);
    
    num_partials = length(partial_reps);
    
    save(outname, 'partial_reps', 'partial_keys', 'partial_widths', ...
        'partial_num_blocks', 'num_partials', 'thresh');
    
else
    partial_reps = {};
    partial_keys = {};
    partial_widths = [];
    partial_num_blocks = [];
    
    num_partials = 0;
    
    save(outname, 'partial_reps', 'partial_keys', 'partial_widths', ...
        'partial_num_blocks', 'num_partials', 'thresh');
    
end
