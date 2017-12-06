function [song_pattern] = stitch_diags(thresh_diags, Z_or_N)

% STITCH_DIAGS stitches the information from THRESH_DIAGS into one row
% (where each entry represents a time step and the given number denotes
% which group that time step is a member of).
%
% WARNING -- THRESH_DIAGS must be symmetric!!!!!!
% 
% INPUT: THRESH_DIAGS -- Binary matrix with 1's at each pair (SI,SJ) and
%                        0's elsewhere. 
%        Z_OR_N -- Flag determining whether we use NAN or 0's in temporary
%                  variables. 0 means use zeros and 1 means use NANs
%
% OUTPUT: SONG_PATTERN -- A row where each entry represents a time step and
%                         the given number denotes which group that time
%                         step is a member of

T = thresh_diags;
n = size(T,1);

if Z_or_N == 0
    P = zeros(1,n);
elseif Z_or_N == 1
    P = NaN(1,n);
end

% Initialize the grouping ID number, called PATTERN_NUM
pattern_num = 1; 

S = sum(T,1); % Sum down the rows of T
% Create a list of all time steps that are contained in the pairs 
%   representing matrix entries with 1's
check_inds = find(S ~= 0);

P_mask = ones(1,n); % Vector is the same length as the song
% Remove all time steps that are in CHECK_INDS list
P_out = (S == 0);
P_mask = P_mask - P_out;

while ~isempty(check_inds) % Loop over all CHECK_INDS list
    i = check_inds(1); % Choose the first time step in CHECK_INDS, called I
    T_temp = T(i,:); % Isolate that row in the matrix 
    
    inds = find(T_temp ~= 0); % Find all time steps in that row that I is 
                              % close to. Call this list INDS
    
    if ~isempty(inds) % If INDS is not empty, proceed
        while ~isempty(inds) % While INDS is not empty, proceed
            
            C_mat = sum(T(inds,:),1).*P_mask; % 1) Isolate the rows of T 
                                              %     that correspond to INDS
                                              % 2) Sum down those rows
                                              % 3) Multiply resulting
                                              %    vector by P_MASK (i.e.
                                              %    set any already assigned
                                              %    time steps to 0)
                                              
            C_inds = find(C_mat ~= 0); % Make a list of time steps that are 
                                       % in the group with I. Making this 
                                       % list relies on transitivity. In 
                                       % other words, there could be an
                                       % element of this list that is not
                                       % detected as being related to I,
                                       % but is detected as being related 
                                       % to an element in INDS, which are 
                                       % all detected as being related to
                                       % I.
            
            P(C_inds) = pattern_num; % Give all elements of C_INDS the same 
                                     % grouping ID number as I
            
            check_inds = setdiff(check_inds,C_inds); % Since the elements 
                                                     % of C_INDS now have
                                                     % grouping ID numbers,
                                                     % we remove them from
                                                     % the CHECK_INDS list
                                                     
            P_mask(C_inds) = 0; % Also remove C_INDS from P_MASK
            
            inds = setdiff(C_inds,inds); % Reset INDS to be C_INDS with 
                                         % INDS removed. Recalling the
                                         % transitivity argument from
                                         % above, this resetting makes
                                         % sense since C_INDS is the union
                                         % of the INDS search on all the
                                         % rows associated with INDS (not
                                         % just the first row, like INDS
                                         % is).
        end
        pattern_num = pattern_num + 1; % Once the while loop is cleared, 
                                       % update the grouping ID number
    end
    check_inds = setdiff(check_inds,i); % Remove I from CHECK_INDS
end

song_pattern = P;

end