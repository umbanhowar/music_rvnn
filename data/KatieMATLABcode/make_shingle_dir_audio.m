function make_shingle_dir_audio(shingleStart,shingleEnd)

% MAKE_SHINGLE_DIR_AUDIO makes directories for each shingle width specified
%
% INPUT: SHINGLESTART -- Smallest number of feature vectors per shingle
%        SHINGLESEND -- Largest number of feature vectors per shingle
%
% OUTPUT: None. Directories are created

for i = shingleStart:shingleEnd
    dirname = ['MusicFiles/MATfiles'];
    mkdir(dirname)
    
    dirnameS = ['MusicFiles/MATfiles/ShingleNumber', num2str(i)];
    mkdir(dirnameS)
    
    for j = 1:4
        dirnameOp = ['MusicFiles/MATfiles/ShingleNumber', num2str(i), ...
            '/mazurka06-', num2str(j)];
        mkdir(dirnameOp)
    end
end