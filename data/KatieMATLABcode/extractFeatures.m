% Based on demoChromaToolbox.m by Mueller and Ewert

addpath /ihome/kinnaird/MATLAB-Chroma-Toolbox_2.0/
addpath /ihome/kinnaird/MATLAB-Chroma-Toolbox_2.0/data_feature/
addpath /ihome/kinnaird/MATLAB-Chroma-Toolbox_2.0/data_WAV/
addpath /ihome/kinnaird/

[f_audio,sideinfo] = wav_to_audio('/ihome/kinnaird/','','tmpfile.wav');
shiftFB = estimateTuning(f_audio);

paramPitch.winLenSTMSP = 4410;
paramPitch.shiftFB = shiftFB;
paramPitch.visualize = 0;
[f_pitch,sideinfo] = audio_to_pitch_via_FB(f_audio,paramPitch,sideinfo);

paramCP.applyLogCompr = 0;
paramCP.visualize = 0;
paramCP.inputFeatureRate = sideinfo.pitch.featureRate;
[f_CP,sideinfo] = pitch_to_chroma(f_pitch,paramCP,sideinfo);
save('/ihome/kinnaird/temp_CP.mat', 'f_CP', 'sideinfo', 'paramCP');

num_FV_per_shingle = 30;
[distAS_CP] = DistMat_from_FeatureVectors(f_CP, num_FV_per_shingle);
save('/ihome/kinnaird/temp_CP_dist.mat', 'distAS_CP', ...
    'num_FV_per_shingle');

paramCLP.applyLogCompr = 1;
paramCLP.factorLogCompr = 100;
paramCLP.visualize = 0;
paramCLP.inputFeatureRate = sideinfo.pitch.featureRate;
[f_CLP,sideinfo] = pitch_to_chroma(f_pitch,paramCLP,sideinfo);
save('/ihome/kinnaird/temp_CLP.mat', 'f_CLP', 'sideinfo', 'paramCLP');

num_FV_per_shingle = 30;
[distAS_CLP] = DistMat_from_FeatureVectors(f_CLP, num_FV_per_shingle);
save('/ihome/kinnaird/temp_CLP_dist.mat', 'distAS_CLP', ...
    'num_FV_per_shingle');

paramCENS.winLenSmooth = 41;
paramCENS.downsampSmooth = 10;
paramCENS.visualize = 0;
paramCENS.inputFeatureRate = sideinfo.pitch.featureRate;
[f_CENS,sideinfo] = pitch_to_CENS(f_pitch,paramCENS,sideinfo);
save('/ihome/kinnaird/temp_CENS.mat', 'f_CENS', 'sideinfo', 'paramCENS');

num_FV_per_shingle = 3;
[distAS_CENS] = DistMat_from_FeatureVectors(f_CENS, num_FV_per_shingle);
save('/ihome/kinnaird/temp_CENS_dist.mat', 'distAS_CENS', ...
    'num_FV_per_shingle');

paramCRP.coeffsToKeep = (55:120);
paramCRP.visualize = 0;
paramCRP.inputFeatureRate = sideinfo.pitch.featureRate;
[f_CRP,sideinfo] = pitch_to_CRP(f_pitch,paramCRP,sideinfo);
save('/ihome/kinnaird/temp_CRP.mat', 'f_CRP', 'sideinfo', 'paramCRP');

num_FV_per_shingle = 30;
[distAS_CRP] = DistMat_from_FeatureVectors(f_CRP, num_FV_per_shingle);
save('/ihome/kinnaird/temp_CRP_dist.mat', 'distAS_CRP', ...
    'num_FV_per_shingle')

paramSmooth.winLenSmooth = 41;
paramSmooth.downsampSmooth = 10;
paramSmooth.inputFeatureRate = sideinfo.CRP.featureRate;
[f_CRPSmoothed, featureRateSmoothed] = ...
    smoothDownsampleFeature(f_CRP,paramSmooth);
save('/ihome/kinnaird/temp_CRPS.mat', 'f_CRPSmoothed', 'sideinfo', ...
'paramSmooth');

num_FV_per_shingle = 3;
[distAS_CRPS] = ...
    DistMat_from_FeatureVectors(f_CRPSmoothed, num_FV_per_shingle);
save('/ihome/kinnaird/temp_CRPS_dist.mat', 'distAS_CRPS', ...
    'num_FV_per_shingle')

rmpath /ihome/kinnaird/MATLAB-Chroma-Toolbox_2.0/
rmpath /ihome/kinnaird/MATLAB-Chroma-Toolbox_2.0/data_feature/
rmpath /ihome/kinnaird/MATLAB-Chroma-Toolbox_2.0/data_WAV/
rmpath /ihome/kinnaird/