function [ylabels] = get_yLabels(width_vec, anno_vec)

% GET_YLABLES generates the labels for a visualization with WIDTH_VEC and
% ANNO_VEC
%
% INPUT: WIDTH_VEC -- Vector of widths for a visualization
%        ANNO_VEC -- Vector of annotations for a visualization
%
% OUTPUT: YLABELS -- Labels for the y-axis of a visualization

num_rows = size(width_vec,1); assert(num_rows == size(anno_vec,1));

ylabels = cell([num_rows,1]);
for i = 1: num_rows
    ylabels{i} = ['w = ', num2str(width_vec(i)), ...
        ', a = ', num2str(anno_vec(i))];
end