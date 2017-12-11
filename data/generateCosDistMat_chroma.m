function [] = generateCosDistMat_chroma(task_id_string)
    task_num = str2num(task_id_string);
    filepaths = importdata('filepaths.txt');
    %175 - bach chor001
    %900 - long one with artifacts
    path = filepaths{task_num};
    matfile_path = strcat(path, '.mat');
    mat = load(matfile_path);
    piano_roll = mat.data;
    
    reduced = zeros(12, size(piano_roll, 2));
    for i=1:size(piano_roll, 1)
        chroma = mod((i - 1), 12) + 1;
        for j=1:size(piano_roll, 2)
            if piano_roll(i, j) > reduced(chroma, j)
                reduced(chroma, j) = piano_roll(i, j);
            end
        end
    end

    zeros_mask = sum(reduced, 1) == 0;
    reduced(:, zeros_mask) = eps;

    [sim,matas] = cosDistMat_from_FeatureVectors_Nate(reduced, 1, 12);
    dlmwrite(strcat(path, '_sim1_12chroma.txt'), sim);
end