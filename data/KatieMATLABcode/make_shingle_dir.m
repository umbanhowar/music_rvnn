function make_shingle_dir(shingleStart,shingleEnd)

% MAKE_SHINGLE_DIR makes directories for each shingle width specified
%
% INPUT: SHINGLESTART -- Smallest number of feature vectors per shingle
%        SHINGLESEND -- Largest number of feature vectors per shingle
%
% OUTPUT: None. Directories are created

for i = shingleStart:shingleEnd
    dirname = ['MATfiles/ShingleNumber', num2str(i)];
    mkdir(dirname)
    
    dirnameE = ['MATfiles/ShingleNumber', num2str(i),'/Expanded'];
    mkdir(dirnameE)
    
    dirnameN = ['MATfiles/ShingleNumber', num2str(i),'/NotExpanded'];
    mkdir(dirnameN)
end