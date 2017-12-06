function [all_lst] = ...
    lightup_lst_with_thresh_bw(thresh_mat, bandwidth_vec, thresh_bw)

% LIGHTUP_LST_WITH_THRESH_BW finds all diagonals present in THRESH_MAT,
% removing each diagonal as it is found.
%
% INPUT: THRESH_MAT -- Thresholded matrix that we extract diagonals from
%        BANDWIDTH_VEC -- Vector of lengths of diagonals to be found
%        THRESH_BW -- Smallest allowed diagonal length
%
% OUTPUT: ALL_LST -- List of pairs of repeats that correspond to diagonals
%                    in THRESH_MAT

% Initialize the input and temporary variables
T_temp = thresh_mat;
Bvec = bandwidth_vec;
b = length(Bvec);
Tbw = thresh_bw;

int_all = [];  % Interval list for non-overlapping pairs
sint_all = []; % Interval list for the left side of the overlapping pairs
eint_all = []; % Interval list for the right side of the overlapping pairs
mint_all = []; % Interval list for the middle of the overlapping pairs if 
               %    they exist
               
for i = 1:b % Loop over all possible bandwidths
    % Set current bandwidth
    j = b-i+1;
    bw = Bvec(j);
    
    if bw > Tbw
        % Search for diagonals of length BW
        DDM = conv2(T_temp(1:end,1:end),eye(bw),'valid');
        % Mark where diagonals of length BW start
        T_DDM = (DDM == bw);
        
        if sum(sum(T_DDM)) > 0
            full_bw = bw;
            % 1) Non-Overlaps: Search outside the overlapping shingles
            [SI, SJ] = find(triu(T_DDM, full_bw)); % Find the starts that  
                                                   % are paired together
            num_nonoverlaps = size(SI,1);
            % Find the matching ends EI for SI and EJ for SJ
            EI = SI + (full_bw - 1);
            EJ = SJ + (full_bw - 1);
            
            % List pairs of starts with their ends and the widths of the
            % non-overlapping intervals
            int_lst = [SI, EI, SJ, EJ, full_bw*ones(num_nonoverlaps,1)];
            
            % Add the new non-overlapping intervals to the full list of
            % non-overlapping intervals
            int_all = [int_lst; int_all];
            
            % 2) Overlaps: Search only the overlaps in shingles
            [SIo, SJo] = find(tril(triu(T_DDM), (full_bw - 1)));
            num_overlaps = size(SIo,1);
            
            if (num_overlaps == 1 && SIo == SJo)
                sint_lst = [SIo, (SIo + (full_bw - 1)), ...
                    SJo, (SJo + (full_bw - 1)), full_bw];
                sint_all = [sint_all; sint_lst];
                
            elseif num_overlaps > 0
                % Since you are checking the overlaps you need to cut these
                % intervals into pieces: left, right, and middle. NOTE: the
                % middle interval may NOT exist
                
                % Vector of 1's that is the length of the number of
                % overlapping intervals. This is used a lot. 
                ones_no = ones(num_overlaps,1);
                
                % 2a) Left Overlap
                K = SJo - SIo; % NOTE: EJO - EIO will also equal this, 
                               % since the intervals that are overlapping 
                               % are the same length. Therefore the "left" 
                               % non-overlapping section is the same length 
                               % as the "right" non-overlapping section. It 
                               % does NOT follow that the "middle" section 
                               % is equal to either the "left" or "right" 
                               % piece. It is possible, but unlikely.
                               
                sint_lst = [SIo, (SJo - ones_no), ...
                    SJo, (SJo + K - ones_no), K];
                [~, Is] = sort(K,'ascend');
                sint_lst = sint_lst(Is, :);
                % Remove the pairs that fall below the bandwidth threshold
                cut_s = find((sint_lst(:,5) > Tbw),1);
                sint_lst = sint_lst(cut_s:end,:);
                
                % Add the new left overlapping intervals to the full list
                % of left overlapping intervals
                sint_all = [sint_all; sint_lst];
                
                % 2b) Right Overlap
                EIo = SIo + (full_bw - 1);
                EJo = SJo + (full_bw - 1);
                eint_lst = [(EIo + ones_no - K), EIo, ...
                                    (EIo + ones_no), EJo, K];
                [~, Ie] = sort(K,'ascend');
                eint_lst = eint_lst(Ie, :);
                
                % Remove the pairs that fall below the bandwidth threshold
                cut_e = find((eint_lst(:,5) > Tbw),1);
                eint_lst = eint_lst(cut_e:end,:);
                
                % Add the new right overlapping intervals to the full list
                % of right overlapping intervals
                eint_all = [eint_all; eint_lst];
                
                % 2) Middle Overlap
                mnds = (EIo - SJo - K + ones_no) > 0;
                SIm = SJo(mnds);
                EIm = (EIo(mnds) - K(mnds));
                SJm = (SJo(mnds) + K(mnds));
                EJm = EIo(mnds);
                Km = (EIo(mnds) - SJo(mnds) - K(mnds) + ones_no(mnds));
                
                if sum(sum(mnds)) > 0
                    mint_lst = [SIm, EIm, SJm, EJm, Km];
                    [~, Im] = sort(Km,'ascend');
                    mint_lst = mint_lst(Im, :);
                    
                    % Remove the pairs that fall below the bandwidth 
                    %   threshold
                    cut_m = find((mint_lst(:,5) > Tbw),1);
                    mint_lst = mint_lst(cut_m:end,:);
                    
                    % Add the new middle overlapping intervals to the full 
                    % list of middle overlapping intervals
                    mint_all = [mint_all; mint_lst];
                end
            end

            % 4) Remove found diagonals of length BW from consideration
            [SDM] = stretch_diags(T_DDM, bw);
            T_temp = T_temp - SDM; 
        end
        
        if sum(sum(T_temp)) == 0
            break
        end
    end
end

% Combine non-overlapping intervals with the left, right, and middle parts
% of the overlapping intervals
overlap_lst = [sint_all; eint_all; mint_all];
all_lst = [int_all; overlap_lst];

% Sort the list by bandwidth size
[~, I] = sort(all_lst(:,5),'ascend');
all_lst = all_lst(I, :);

end