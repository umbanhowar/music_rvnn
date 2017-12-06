function file2HR_save(inname, outname, num_FV_per_shingle, thresh)

% FILE2PARTIALS_SAVE processes a specific file from the original feature
% vectors to the hierarchical representation, saving it in OUTNAME.
%
% INPUT: INNAME -- Name of file to be processed
%        OUTNAME -- Name of file where partials will be saved
%        NUM_FV_PER_SHINGLE -- Number of feature vectors per shingle
%        THRESH -- Maximum threshold value
%
% OUTPUT: None. File is saved.

% Import file of feature vectors
Fmat = importdata(inname);

% Get pairwise distance matrix using ``cosine distance''
[distAS] = cosDistMat_from_FeatureVectors(Fmat, num_FV_per_shingle);

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
    
    % We save a list of partial representations that only contains the full
    % hierarchical representation to be able to use the comparison code
    partial_reps = {full_matrix_no};
    partial_keys = {full_key};
    partial_widths = [sn];
    partial_num_blocks = sum(sum(matrix_no));
    num_partials = 1;
    
    save(outname, 'full_key', 'full_matrix_no', 'partial_reps', ...
        'partial_keys', 'partial_widths', 'partial_num_blocks', ...
        'num_partials','thresh');
    
else
    full_key = [];
    full_matrix_no = [];
    
    % We save an empty list of partial representations to be able to use
    % the comparison code
    
    partial_reps = {};
    partial_keys = {};
    partial_widths = [];
    partial_num_blocks = [];
    
    num_partials = 0;
    
    save(outname, 'full_key', 'full_matrix_no', 'partial_reps', ...
        'partial_keys', 'partial_widths', 'partial_num_blocks', ...
        'num_partials','thresh');
    
end
end
