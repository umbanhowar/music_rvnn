function [precision_score, recall_score] = ...
    precision_recall_values(Adjmat, correct_mat)

% PRECISION_RECALL_VALUES computes the precision and recall values for a
% given adjacency ADJMAT where CORRECT_INDS states the correct index pairs.
%
% INPUT: ADJMAT -- Adjaceny matrix 
%        CORRECT_MAT -- Binary matrix of correct matches. We define that
%                       CORRECT_MAT(i,j) = 1 if i-th data point is matched
%                       to the j-th data point.
%
% OUTPUT: PRECISION_SCORE -- Precision score for ADJMAT given CORRECT_MAT
%         RECALL_SCORE -- Recall score for ADJMAT given CORRECT_MAT

A = Adjmat;
a = size(A,1);
num_matches_A = sum(sum(A));

CM = correct_mat;
num_matches_C = sum(sum(CM));

Found_CM_in_A = sum(sum(((A + CM) == 2)));

% Precision is number of matches in A and CI divided by number of matches
% in just A (see \cite{KK2007})
precision_score = Found_CM_in_A/num_matches_A;

% Recall is number of matches in A and CI divided by number of matches
% in just CI (see \cite{KK2007})
recall_score = Found_CM_in_A/num_matches_C;