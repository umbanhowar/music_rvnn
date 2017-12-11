filepaths = importdata('filepaths.txt');

for ii=1:1001
    path = filepaths{ii};
    mat_path = strcat(path, '_sim1_12chroma.txt');
    try
        dlmread(mat_path);
    catch
        ii
        generateCosDistMat_chroma(num2str(ii));
        'done'
    end
end