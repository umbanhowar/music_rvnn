function [] = generateCosDistMat(task_id_string)
    task_num = str2num(task_id_string);

    filepaths = importdata('filepaths.txt');
    %175 - bach chor001
    %900 - long one with artifacts
    path = filepaths{task_num};
    matfile_path = strcat(path, '.mat');
    mat = load(matfile_path);
    piano_roll = mat.data;

    zeros_mask = sum(piano_roll, 1) == 0;
    piano_roll(:, zeros_mask) = eps;

    [sim,matas] = cosDistMat_from_FeatureVectors_Nate(piano_roll, 3, 4);
    dlmwrite(strcat(path, '_sim3_4full.txt'), sim);
end