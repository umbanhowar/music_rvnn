% Set thresh each time
thresh = 0.05;
load('/Users/katiek/Desktop/CurrentCode/MATfiles/HRmat.mat')

% Loop over s = 6 and s = 12
for s = [6,12]
    make_shingle_dir(s,s)
    % Process files
    tic;process_dir2HR_save(s, thresh, 1);toc;
    % Get pairwise distance matrix
    tic;dist_mat = get_whole_set_dist(3,s,1);toc;
    % Get results for run
    [adj_mat, num_empty_reps, num_poss_matches, ...
        precision_score, recall_score] = ...
        run_results(dist_mat, 'D', 'Z', 'S', HR_Correct_mat, s);
    % Save graph
    adj_save_name = ['/Users/katiek/Desktop/CurrentCode/MATfiles/'...
        'ShingleNumber', num2str(s),'/adjmat.mat'];
    save(adj_save_name, 'adj_mat');
    
    fprintf(['For shingle width ',num2str(s),', we have ', ...
        num2str(num_poss_matches), ' possible matches with precision ', ...
        num2str(precision_score), ' and with recall ', ...
        num2str(recall_score), '. There are ', num2str(num_empty_reps), ...
        ' empty representations. \n']);
    
    clear dist_mat; clear adj_mat; clear adj_save_name; clear recall_score;
    clear num_empty_reps; clear num_poss_matches; clear precision_score;
end