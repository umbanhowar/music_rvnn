function partial_lst_distmat = compare_partial_lst(compare_songlist)

% COMPARE_PARTIAL_LIST computes the matrix of pairwise comparisons between
% the representations (or lists of representations) listed in
% COMPARE_SONGLIST. For this comparison, we count the number of exactly
% matching representations in file I and file J and dividing that count by
% the total number of representations in file I and in file J.
%
% INPUT: COMPARE_SONGLIST -- List of filenames for the representations (or
%                            lists of representations) that are to be
%                            compared
%
% OUTPUT: PARTIAL_LST_DISTMAT -- Matrix of pairwise comparisons between
%                                the representations (or lists of
%                                representations) listed in
%                                COMPARE_SONGLIST

num_songs = length(compare_songlist);
D = zeros(num_songs);

for i = 1:(num_songs - 1)
    % Load the I-th file
    file_i = compare_songlist{i};
    songI = load(file_i);
    
    I_partial_reps = songI.partial_reps;
    I_partial_widths = songI.partial_widths;
    I_partial_keys = songI.partial_keys;
    I_partial_num_blocks = songI.partial_num_blocks;
    I_num_reps = songI.num_partials;
    
    % If song I has no partial representations then D(I,J) = 0 for all J
    if (I_num_reps > 0)
        for j = (i + 1): num_songs
            % Load the J-th file
            file_j = compare_songlist{j};
            songJ = load(file_j);
            
            J_partial_reps = songJ.partial_reps;
            J_partial_widths = songJ.partial_widths;
            J_partial_keys = songJ.partial_keys;
            J_partial_num_blocks = songJ.partial_num_blocks;
            J_num_reps = songJ.num_partials;
            
            % Start the distance computation
            
            % If Song J has no partials, then D(i,j) = 0. Only proceed if
            % Song J has partials.
            if (J_num_reps > 0)
                I_key_mat = repmat(I_partial_widths', 1, J_num_reps);
                J_key_mat = repmat(J_partial_widths, I_num_reps, 1);
                find_compare_points = (I_key_mat == J_key_mat);
                
                I_blocknum_mat = repmat(I_partial_num_blocks', 1, J_num_reps);
                J_blocknum_mat = repmat(J_partial_num_blocks, I_num_reps, 1);
                
                D_temp = (I_blocknum_mat + J_blocknum_mat).*...
                    (ones(I_num_reps, J_num_reps) - find_compare_points);
                
                [I_inds,J_inds] = find(D_temp == 0);
                % If I_inds is empty, then we have no partials to compare
                % and D(i,j) = 0.
                if ~isempty(I_inds)
                    for k = 1:size(I_inds,1)
                        partialrepI = I_partial_reps{I_inds(k)};
                        partialkeyI = I_partial_keys{I_inds(k)};
                        
                        partialrepJ = J_partial_reps{J_inds(k)};
                        partialkeyJ = J_partial_keys{J_inds(k)};
                        D_temp(I_inds(k),J_inds(k)) = ...
                            dist_between_reps(partialrepI, partialkeyI,...
                            partialrepJ, partialkeyJ);
                        
                    end
                    [I_exact, J_exact] = find(D_temp == 0);
                    
                    D(i,j) = (2*size(I_exact,1))/(I_num_reps + J_num_reps);
                end
            end
        end
    end
end

partial_lst_distmat = D + D';