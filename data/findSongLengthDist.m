filepaths = importdata('filepaths.txt');

lengths = zeros(1, length(filepaths));
for i=1:length(filepaths)
    i
    sim_mat = dlmread(strcat(filepaths{i}, '_sim14full.txt'));
    lengths(i) = size(sim_mat, 1);
end