function [out_partial_reps, out_partial_keys, out_partial_widths, ...
    out_partial_num_blocks] = find_unique_partial_reps(in_partial_reps, ...
    in_partial_keys, in_partial_widths, in_partial_num_blocks)

% FIND_UNIQUE_PARTIAL_REPS removes all copies of partial representations
% leaving a list of unique partial representations. 
%
% INPUT: IN_PARTIAL_REPS -- List of partial hierarchical representations
%                           for the hierarchical representation stored in
%                           FULL_VIS (and FULL_VIS_KEY)
%        IN_PARTIAL_KEYS -- List of vectors of lengths of repeats in the
%                           partial representations. Each vector of lengths
%                           corresponds to a partial hierarchical
%                           representation stored in IN_PARTIAL_REPS
%        IN_PARTIAL_WIDTHS -- List of the widths corresponding to the
%                             partial hierarchical representations stored
%                             in IN_PARTIAL_REPS
%        IN_PARTIAL_NUM_BLOCKS -- Number of repeats in each partial
%                                 representation in IN_PARTIAL_REPS
%
% OUTPUT: OUT_PARTIAL_REPS -- List of unique partial hierarchical
%                             representations based on partial hierarchical 
%                             representations listed in IN_PARTIAL_REPS
%         OUT_PARTIAL_KEYS -- List of vectors of lengths of repeats in the
%                             partial representations. Each vector of
%                             lengths corresponds to a partial hierarchical
%                             representation stored in OUT_PARTIAL_REPS
%         OUT_PARTIAL_WIDTHS -- List of the widths corresponding to the 
%                               partial hierarchical representations stored 
%                               in OUT_PARTIAL_REPS
%         OUT_PARTIAL_NUM_BLOCKS -- Number of repeats in each partial
%                                   representation in OUT_PARTIAL_REPS

num_reps = length(in_partial_reps);

key_mat = repmat(in_partial_widths', 1, num_reps);

% Find unique pairs of representations of the same width in the list
% IN_PARTIAL_REPS. Here we choose to keep pairs (i,j) with i<j.
[I_inds,J_inds] = find(triu(key_mat == key_mat',1));

% Compute the distance between every pair of partial hierarchical
% representations of the same width and remove any copies, always keeping
% one from any pairs with distance 0
remove_inds = [];
if ~isempty(I_inds)
    for i = 1:size(I_inds,1)
        partialrepI = in_partial_reps{I_inds(i)};
        partialkeyI = in_partial_keys{I_inds(i)};
        
        partialrepJ = in_partial_reps{J_inds(i)};
        partialkeyJ = in_partial_keys{J_inds(i)};
        DIJ = dist_between_reps(partialrepI, partialkeyI,...
                                                partialrepJ, partialkeyJ);             
        if DIJ == 0
            remove_inds = [remove_inds, J_inds(i)];
        end
    end
end

% Keep the first instance of each unique partial hierarchical
% representation
keep_inds = setdiff([1:num_reps],remove_inds);

out_partial_reps = in_partial_reps(keep_inds);
out_partial_keys = in_partial_keys(keep_inds);
out_partial_widths = in_partial_widths(keep_inds);
out_partial_num_blocks = in_partial_num_blocks(keep_inds);
