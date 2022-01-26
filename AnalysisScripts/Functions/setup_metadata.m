function [sets] = setup_metadata()
% setup_metadata: Setup the settings which control how we'll run all decoding analyses
%   Inputs:
%       N/A
%
%   Outputs:
%       set - Structure describing how the decoding analysis will run. Structure is as follows:
%             direct: directories for data and results
%                str: strings describing conditions
%                  n: counts for conditions
%                eeg: eeg recording settings
%             timing: timing for sliding windows etc.
%              epoch_chunk: epoching settings for sliding window chunks
%         epoch_long: epoching settings for full trial
%                 Hz: frequency tagging analysis settings
%             netset: settings for neural network
%               trig: triggers used in EEG data. 

%% Base Directories
% data
direct.main = '..\'; %change this if you would like to store data in a separate repository, e.g. 'D:\FeatAttnClassification\';
direct.data = [direct.main 'Data\'];

% results
direct.results = '..\Results\' ; 
direct.results_group = '..\Results\Group\' ; 

% toolboxes
direct.fieldtrip = 'C:\Users\uqarento\Documents\toolboxes\fieldtrip-20180422\';
addpath(direct.fieldtrip);

% set settings
sets.direct = direct;

%% Strings

str.HzState = {'Basic' 'Harmonic' 'Alpha' 'AlphaPlusHarmonic'};
str.testtrainopts = {'MultiFreq' 'SingleFreq'};
str.trainstrings = {'TrainMultiFreqTestMultiFreq' 'TrainSingleFreqTestMultiFreq'};
str.excludemotepochs = {'' 'include_motepoch'};
str.colcond = {'B:6Hz, W:7.5Hz', 'W:6Hz, B:7.5Hz'};

str.col = {'Black' 'White'};
str.Hz = {'6Hz' '7.5Hz'};
str.Hzattend = {'Attend 6Hz' 'Attend 7.5 Hz'};

str.methods = {'MLP' 'LDA'  'KNN' 'SVM' 'zscore'  'LR_L2'}; %'LR_L1',

% set settings
sets.str = str;

%% Basic count settings for task
n.stimstates = 2;
n.cols = 2;
n.dirs = 4;

n.trials_all = 160;
n.trials_cond = n.trials_all/n.stimstates;
n.motionevents = 5;

% Subject counts
n.subs = 30;
n.sub_ids = 32;

% decoding count settings
n.methods = length(str.methods);
n.hzstates = 4;
n.folds = 10;
sets.fraction_motexclude = 1/3;
n.traintypes = 2;

% set settings
sets.n = n;

%% EEG settings 
eeg.n_chans = 5;
eeg.fs = 1200;

% set settings
sets.eeg = eeg;

%% Timing settings

time.trial = 15; %seconds
time.slidingwindowstep =0.25; 
time.chunksizes = [0.25 0.5 1 2 4];
time.chunksizes_padded = [2 2 2 2 4];
time.motionepochs = 1;
sets.n.chunksizes = length(time.chunksizes);

% Transform to sample space
tmp = fields(time);
for ii = 1:length(tmp)
    samples.(tmp{ii}) = time.(tmp{ii})*eeg.fs;
end

% set settings
sets.timing.secs = time;
sets.timing.samples = samples;

%% epoch extraction settings (chunks)
% epochs for all chunk sizes get zero-padded out to the same length. 

for chunk = 1:sets.n.chunksizes
    sets.epoch_chunk{chunk} = calc_timingvars([0 time.chunksizes_padded(chunk)]);
end

%% epoch extraction settings (full trial)
sets.epoch_long = calc_timingvars([0 15]);

%% define nested function to calculate timing settings

    function [out] = calc_timingvars(lim_s)
        out.lim.s = lim_s;
        out.lim.x = out.lim.s.*eeg.fs;
        
        out.n.s = out.lim.s(2) - out.lim.s(1);
        out.n.x = out.lim.x(2) - out.lim.x(1);
        
        out.lim.x(1) = out.lim.x(1) + 1;
        
        out.t = out.lim.s(1) : 1/eeg.fs : out.lim.s(2) - 1/eeg.fs;
        out.f = 0 : 1/out.n.s : eeg.fs - 1/out.n.s;
    end

%% Frequency settings
Hz.f1 = [6 7.5];
Hz.f2 = Hz.f1*2;
Hz.alpha_bounds = [8 12];

% vector lengths
sets.n.Hz_f1 = length(Hz.f1); 
sets.n.Hz_f2 = length(Hz.f2); 

for chunk = 1:sets.n.chunksizes
    % settings for chunks - f1
    [Hz.f1_idx{chunk}, Hz.f1_real{chunk}]  = calc_hzidx(Hz.f1, sets.epoch_chunk{chunk}.f);
    
    % settings for chunks - f2
    [Hz.f2_idx{chunk}, Hz.f2_real{chunk}]  = calc_hzidx(Hz.f2, sets.epoch_chunk{chunk}.f);
    
    % Settings for chunks - alpha
    [tmp_alpha,~]  = calc_hzidx(Hz.alpha_bounds, sets.epoch_chunk{chunk}.f);
    Hz.alpha_idx{chunk} = tmp_alpha(1):tmp_alpha(2);
    Hz.alpha_real{chunk} = sets.epoch_chunk{chunk}.f(Hz.alpha_idx{chunk});
    sets.n.Hz_alpha{chunk} = length(Hz.alpha_real{chunk});
    
end
% Settings for full trial - f1
[Hz.epoch_long.f1_idx, Hz.epoch_long.f1_real]  = calc_hzidx(Hz.f1, sets.epoch_long.f);

% set settings
sets.Hz = Hz;

%% define nested function to calculate index of freq in var f, and the real
% value of that frequency (i.e. if we don't have the spectral resolution
% needed, our freq will be shifted. 
    function [hz_idx, hz_real] = calc_hzidx(hz, f)
        hz_idx = NaN(1,length(hz));
        for ii_Hz = 1:length(hz)
            [~,  hz_idx(ii_Hz)] = min(abs(f-hz(ii_Hz)));
        end
        hz_real = f(hz_idx);
    end

%% Neural Network Settings

%## Choose a Training Function
% For a list of all training functions type: help nntrain
% 'trainlm' is usually fastest.
% 'trainbr' takes longer but may be better for challenging problems. Bayesian
% 'trainscg' uses less memory. Suitable in low memory situations. Scaled conjugate gradient backpropagation.
netset.trainFcn = 'traingdx';% like ADAM - has momentum

%## Choose a Performance
% For a list of all performance functions type: help nnperformance
netset.performFcn = 'mse';%'crossentropy';

%## Create a Pattern Recognition Network
% hiddenLayerSize = [20 4];%
netset.hiddenLayerSize = [10 10];%

%## Setup Division of Data for Training, Validation, Testing
% For a list of all data division functions type: help nndivide
netset.divideFcn = 'dividerand';  % Divide data randomly
netset.divideMode = 'sample';  % Divide up every sample
netset.divideParam.trainRatio = 70/100;
netset.divideParam.valRatio = 30/100;
netset.divideParam.testRatio = 0/100;

%## For a list of all plot functions type: help nnplot
netset.plotFcns = {'plotperform','plottrainstate','ploterrhist', ...
    'plotconfusion', 'plotroc'};

% set settings
sets.netset = netset;


%% Trigger settings 
% These are the triggers embedded in the data

% Cue Start
% Hz Attend, Col Attend, Training Cond
trig.cue = NaN(sets.n.Hz_f1,n.cols,n.stimstates);

trig.cue(1,:,:) = [
    1 2 ;
    3 4    ];

trig.cue(2,:,:) = [
    5 6;
    7 8    ];

% Trial Start
% Hz Attend, Col Attend, Training Cond
trig.trial = trig.cue + 20;

% Motion Onset
% Hz Attd move, Col move, # of  dot fields, motion direction
trig.motion = NaN(sets.n.Hz_f1,n.cols,n.stimstates, n.dirs);

trig.motion(1,1,:,:) = [
    101 102 103 104;
    105 106 107 108
    ];

trig.motion(1,2,:,:) = trig.motion(1,1,:,:)+8;
trig.motion(2,1,:,:) = trig.motion(1,2,:,:)+8;
trig.motion(2,2,:,:) = trig.motion(2,1,:,:)+8;

% Feedback start
trig.feedback = 222;

% Set settings
sets.trig = trig;
end
