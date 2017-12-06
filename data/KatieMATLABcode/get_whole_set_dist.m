function out_mat = get_whole_set_dist(comp_flag, num_shingles, save_flag)

% GET_WHOLE_SET_DIST computes the matrix of pairwise comparisons between
% the representations (or lists of representations) in a directory.
%
% INPUT: COMP_FLAG -- Flag that determines which kind of comparison that
%                     is performed. For two representations (or lists of
%                     representations) I and J:
%                       1 means counting the number of exactly matching 
%                               representations in I and J and dividing
%                               that count by the total number of
%                               representations in I and in J
%                       2 means counting the number of exactly matching 
%                               representations in I and J and dividing
%                               that count by the number of representations
%                               in I
%                       3 means adding the distances between
%                               representations of the same width 
% INPUT: NUM_SHINGLES -- The number of feature vectors per shingle
%        SAVE_FLAG -- Flag that determines if the matrix of pairwise
%                     comparisons is saved or not. 0 (default) means that
%                     we do not save the matrix and 1 means that we do save
%                     the matrix
% 
% OUTPUT: OUT_MAT -- Matrix of pairwise comparisons (with the kind of
%                    comparison prescribed by COMP_FLAG) between
%                    representations in the directory.



if nargin < 3
    save_flag = 0;
end

key_dir = '/Users/katiek/Desktop/CurrentCode/FileKeys/';
compare_songlist = importdata([key_dir, ...
    'filenameMAT_list_ShingleNumber', num2str(num_shingles),'.csv']);

% Count the number of exactly matching representations in I and J and
% divide that count by the total number of representations in I and in J
if comp_flag == 1
    comp_mat = compare_partial_lst(compare_songlist);
    if save_flag == 1
        outname = ['/Users/katiek/Desktop/CurrentCode/MATfiles/',...
            'ShingleNumber', num2str(num_shingles),'/compmat.mat'];
        save(outname, 'comp_mat'); 
    end
    out_mat = comp_mat;

% Count the number of exactly matching representations in I and J and
% divide that count by the total number of representations in I   
elseif comp_flag == 2
    directed_comp_mat = compare_partial_lst_directed(compare_songlist);
    if save_flag == 1
        outname = ['/Users/katiek/Desktop/CurrentCode/MATfiles/',...
            'ShingleNumber', num2str(num_shingles),'/directedcompmat.mat'];
        save(outname, 'directed_comp_mat');   
    end
    out_mat = directed_comp_mat;
    
% Add the distances between representations of the same width     
elseif comp_flag == 3
    dist_mat = dist_partial_lst(compare_songlist);
    if save_flag == 1
        outname = ['/Users/katiek/Desktop/CurrentCode/MATfiles/',...
            'ShingleNumber', num2str(num_shingles),'/distmat.mat'];
        save(outname, 'dist_mat');
    end
    out_mat = dist_mat;
end


end