filepaths = importdata('filepaths.txt');

num_mats_to_read = 1001;
indices = randi([1 1001], 1, num_mats_to_read);

all_samples = [];

for i=1:num_mats_to_read
    i
    sim_mat = dlmread(strcat(filepaths{indices(i)}, '_sim1_12full.txt'));
    sz = size(sim_mat, 1);
    num_to_sample = 4*sz;
    samples = datasample(sim_mat(:), num_to_sample, 'Replace', false);
    all_samples = [all_samples; samples];

    %figure();
    %imagesc(sim_mat < 0.25);
end

figure();
h = histogram(all_samples, 'Normalization', 'cdf');

sorted = sort(all_samples);
pct_thresh = 0.05;
t_idx = round(pct_thresh * length(all_samples));
thresh = sorted(t_idx)




% cdf = h.Values;
% 
% pct_thresh = 0.10;
% for i=1:length(cdf)
%     if cdf(i) >= pct_thresh
%         break
%     end
% end
% 
% threshold = h.BinEdges(i);

