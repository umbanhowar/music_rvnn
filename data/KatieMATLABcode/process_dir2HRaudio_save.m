function process_dir2HRaudio_save(num_shingles, thresh, print_flag)

% PROCESS_DIR2HRaudio_SAVE processes directories of files based on audio
% for NUM_SHINGLES, a specific number of feature vectors per shingle, given
% THRESH, a particular threshold.
%
% INPUT: NUM_SHINGLES -- Number of feature vectors per shingle
%        THRESH -- Maximum threshold value
%        PRINT_FLAG -- Flag that determines if the file number that is
%                      being processed gets printed. 0 (default) means the
%                      file numbers will be not printed and 1 means that
%                      the file numbers will be printed
%
% OUTPUT: None. Files are saved.

if nargin < 3
    print_flag = 0;
end

key_dir = '/Users/katiek/Desktop/CurrentCode/FileKeys/';
in_file_lst = importdata([key_dir,'filenameAUDIO_list.csv']);
out_file_lst = importdata([key_dir, 'filenameAUDIO_list_ShingleNumber', ...
    num2str(num_shingles),'.csv']);

for i = 1:length(out_file_lst)
    inname = in_file_lst{i};
    outname = out_file_lst{i};
    
    if print_flag == 1
        fprintf(['Processing File:', inname, '\n']);
    end
    
    file2HR_save_audio(inname, outname, num_shingles, thresh)
end
