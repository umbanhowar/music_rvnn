function [adj_mat, num_empty_reps, num_poss_matches, ...
    precision_score, recall_score] = run_results(comparison_matrix, ...
    comparison_type, adj_mat_flag, data_type, correct_matrix, num_shingles)

% RUN_RESULTS computes the precision and recall values for a given
% COMPARISON_MATRIX computed during a given run of a set of data. 
% 
% INPUT: COMPARISON_MATRIX -- Similarity or dissimilarity matrix resulting
%                             from a given run of a set of data. 
% 	     COMPARISON_TYPE -- Flag stating if COMPARISON_MATRIX is a
%                           similarity or dissimilarity matrix. 'S' means
%                           similarity matrix and 'D' means dissimilarity
%                           matrix
%        ADJ_MAT_FLAG -- Flag stating how we construct the adjacency
%                        matrix. 'Z' means that we only consider the
%                        extreme value of COMPARISON_MATRIX (0 for a
%                        dissimilarity matrix and 1 for a similarity
%                        matrix) and 'K' means we build the mutual 1-NN
%                        graph. 
%        DATA_TYPE -- Flag for the kind of data we are checking. 
%                      'A' means audio and 'S' (default) means score
%        CORRECT_MATRIX -- Binary matrix of correct matches. We define that
%                          CORRECT_MAT(i,j) = 1 if i-th data point is
%                          matched to the j-th data point.
%        NUM_SHINGLES -- Number of feature vectors per shingle
%
% OUTPUT: ADJ_MAT -- The resulting adjacency matrix given COMPARISON_MATRIX
%                    and ADJ_MAT_FLAG 
%         NUM_EMPTY_REPS -- Number of files without any partial
%                           representations
%         NUM_POSS_MATCHES -- Number of possible matches after removing the
%                             data points without representations from 
%                             consideration
%         PRECISION_SCORE -- Precision score for ADJ_MAT given
%                            CORRECT_MATRIX adjusted by removing the data
%                            points without representations from
%                            consideration
%         RECALL_SCORE -- Recall score for ADJ_MAT given CORRECT_MATRIX 
%                         adjusted by removing the data points without
%                         representations from consideration

num_datapoints = size(comparison_matrix,1); 
    assert(num_datapoints == size(comparison_matrix,2));   

% Adjust COMPARISON_MATRIX according COMPARISON_TYPE and ADJ_MAT_FLAG 
if strcmp(comparison_type,'D') && strcmp(adj_mat_flag,'Z')
    myeps = 10^(-14);
    adj_mat = (abs(comparison_matrix)) < myeps;
    
elseif strcmp(comparison_type,'D') && strcmp(adj_mat_flag,'K')
    [min_mat] = dist2minmat(comparison_matrix);
    adj_mat = ((min_mat + min_mat') == 2);
   
elseif strcmp(comparison_type,'S') && strcmp(adj_mat_flag,'Z') 
    myeps = 10^(-14);
    max_val = max(max(comparison_matrix)); 
    adj_mat = (abs((comparison_matrix - max_val)) < myeps);
    
elseif strcmp(comparison_type,'S') && strcmp(adj_mat_flag,'K')  
    [max_mat] = comp2maxmat(comparison_matrix);
    adj_mat = ((max_mat + max_mat') == 2);
end

% Find all the data without any representations
[num_empty_reps, empty_inds] = find_empty_reps(num_shingles, data_type);

% Adjust CORRECT_MATRIX to exclude all rows and columns associated to data
% without any representations
ACM = correct_matrix;
ACM(empty_inds,:) = zeros(num_empty_reps, num_datapoints);
ACM(:,empty_inds) = zeros(num_datapoints, num_empty_reps);

% Find number of possible matches given ACM, the adjusted CORRECT_MATRIX
num_poss_matches = sum(sum(ACM));

[precision_score, recall_score] = precision_recall_values(adj_mat, ACM);