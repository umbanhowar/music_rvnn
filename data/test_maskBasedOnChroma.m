% 0.38 = 10% threshold of 1,4 chroma matrices

filepaths = importdata('filepaths.txt');

num_mats_to_read = 5;
indices = randi([1 1001], 1, num_mats_to_read);
%indices = [175];

all_samples = [];

for i=1:num_mats_to_read
    i
    sim_mat_chroma = dlmread(strcat(filepaths{indices(i)}, '_sim14chroma.txt'));
    sim_mat_full = dlmread(strcat(filepaths{indices(i)}, '_sim14full.txt'));
    
    mask = (sim_mat_chroma < 0.38);
    
    full_masked = mask .* sim_mat_full;
    flip_mask = -1 * (ones(size(mask)) - mask);
    
    figure
    fmc = flip_mask + full_masked;
    imagesc((fmc >= 0) & (fmc < 0.2));
    
    tmp = sim_mat_full(mask);
    samples = tmp;
    all_samples = [all_samples; samples];
end

figure
histogram(all_samples);

sorted = sort(all_samples(:));
pct_thresh = 0.10;
t_idx = round(pct_thresh * length(all_samples));
thresh = sorted(t_idx)