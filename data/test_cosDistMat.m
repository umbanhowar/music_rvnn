f = load('chor001_expanded.mat');
mat = f.data;
reduced = zeros(12, size(mat, 2));
for i=1:128
    chroma = mod((i - 1), 12) + 1;
    for j=1:size(mat, 2)
        if mat(i, j) > reduced(chroma, j)
            reduced(chroma, j) = mat(i, j);
        end
    end
end


[sim,matas] = cosDistMat_from_FeatureVectors_Nate(mat, 1, 4);

imagesc(sim);
%imshow(sim, 'InitialMagnification',3000);