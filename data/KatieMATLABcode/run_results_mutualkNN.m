function [adj_mat, num_empty_reps, num_poss_matches, precision_score, ...
    recall_score] = run_results_mutualkNN(distance_matrix, ...
    num_neighbors, data_type, correct_matrix, num_shingles)

% RUN_RESULTS_MUTUALKNN computes the precision and recall values for a 
% given DISTANCE_MATRIX computed during a given run of a set of data. 
% 
% INPUT: DISTANCE_MATRIX -- Distance matrix resulting from a given run of a
%                           set of data. 
%        NUM_NEIGHBORS -- Number of nearest neighbors
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

num_datapoints = size(distance_matrix,1); 
    assert(num_datapoints == size(distance_matrix,2));   

% Compute mutual kNNG of DISTANCE_MATRIX given NUM_NEIGHBORS
[kNNG_mat] = dist2kNNG(distance_matrix, num_neighbors);
adj_mat = ((kNNG_mat + kNNG_mat') == 2);

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