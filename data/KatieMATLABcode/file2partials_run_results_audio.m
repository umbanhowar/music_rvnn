% Set thresh each time
thresh = 0.10;

% Loop over s = 6 and s = 12
for s = [12]
    make_shingle_dir_audio(s,s)
    % Process files
    tic;process_dir2partials_audio_save(s, thresh, 1);toc;
    % Get pairwise distance matrix
    tic;dist_mat = get_whole_audioset_dist(3,s,1);toc;
    
    clear dist_mat;
end
% Get results for run with 1-kNNG, 10-kNNG, 20-kNNG, 30-kNNG, 40-kNNG, and
% 50-kNNG
run_results_kNNG_loop