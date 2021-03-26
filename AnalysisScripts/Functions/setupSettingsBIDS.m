%% Directories

if ~options.collate % same function used in collate script, which used different directories.
    str.sub = ['S' num2str(SUB)];
    
    if SUB <10
        bids.substring = ['sub-0' num2str(SUB)];
    else
        bids.substring = ['sub-' num2str(SUB)];
    end
    
    %% Setup Directories
    
    direct.data = ['..\data\BIDS\' bids.substring '\eeg\'];
    direct.results = ['..\Results\' str.sub '\']; 
    if ~exist(direct.results, 'dir'); mkdir(direct.results); end
    
end

%% Standard settings

str.HzState = {'Basic' 'Harmonic' 'Alpha' 'AlphaPlusHarmonic'};
str.attnstate = {'MultiFreq' 'SingleFreq' 'SingletoMulti '};

str.Hz = {'6Hz' '7.5Hz'};
str.Hzattend = {'Attend 6Hz' 'Attend 7.5 Hz'};

% Decoding Methods
methods = {'zscore' 'LDA' 'KNN' 'MLP' 'CCA'}; 
n.methods = length(methods);

% EEG Chanels
n.chans = 5;

% Timing settings
time.trial = 15; %seconds
% time.chunksizes = [0.25 0.5 1 1.5 2 4];
time.chunksizes = [0.25 0.5 1 2 4];
time.slidingwindow =0.25; 

% Transform timing settings to sample settings. 
fs = 1200; % samplingrate of EEG

tmp = fields(time);
for ii = 1:length(tmp)
    samples.(tmp{ii}) = time.(tmp{ii})*fs;
end

% Sample settings
n.chunksizes = length(time.chunksizes);

% Timing settings
lim.s = [0 4];
lim.x = lim.s.*fs;

n.s = lim.s(2) - lim.s(1);
n.x = lim.x(2) - lim.x(1);

lim.x(1) = lim.x(1) + 1;

t = lim.s(1) : 1/fs : lim.s(2) - 1/fs;
f = 0 : 1/n.s : fs - 1/n.s;


% Timing settings - long
lim.s_long = [0 15];
lim.x_long = lim.s_long.*fs;

n.s_long = lim.s_long(2) - lim.s_long(1);
n.x_long = lim.x_long(2) - lim.x_long(1);

lim.x_long(1) = lim.x_long(1) + 1;

t_long = lim.s_long(1) : 1/fs : lim.s_long(2) - 1/fs;
f_long = 0 : 1/n.s_long : fs - 1/n.s_long;


% Frequency settings
Hz = [6 7.5 12 15];%12 15];
n.Hz = length(Hz); n.Hz_main = 2;
for HH = 1:n.Hz
    [~, idx.Hz(HH)] = min(abs(f-Hz(HH)));
    Hz_real(HH) = f(idx.Hz(HH));
end

for HH = 1:n.Hz_main
    [~, idx.Hz_long(HH)] = min(abs(f_long-Hz(HH)));
    Hz_real_long(HH) = f_long(idx.Hz(HH));
end

Hz_alpha = [8 12];%12 15];
for HH = 1:2
    [~, tmp_alpha(HH)] = min(abs(f-Hz_alpha(HH)));
end
idx.Hz_alpha = tmp_alpha(1):tmp_alpha(2);

%% Neural Network Settings

%## Choose a Training Function
% For a list of all training functions type: help nntrain
% 'trainlm' is usually fastest.
% 'trainbr' takes longer but may be better for challenging problems. Bayesian
% 'trainscg' uses less memory. Suitable in low memory situations.Scaled conjugate gradient backpropagation.
trainFcn = 'trainscg';

%## Choose a Performance
% For a list of all performance functions type: help nnperformance
performFcn = 'crossentropy';

%## Create a Pattern Recognition Network
% hiddenLayerSize = [20 4];%
hiddenLayerSize = [20 4];%

%## Setup Division of Data for Training, Validation, Testing
% For a list of all data division functions type: help nndivide
net.divideFcn = 'dividerand';  % Divide data randomly
net.divideMode = 'sample';  % Divide up every sample
net.divideParam.trainRatio = 100/100;
net.divideParam.valRatio = 10/100;
net.divideParam.testRatio = 0/100;

%## For a list of all plot functions type: help nnplot
net.plotFcns = {'plotperform','plottrainstate','ploterrhist', ...
    'plotconfusion', 'plotroc'};

%% Trigger settings

n.attnstates = 2;
n.cols = 2;
n.dirs = 4;
% Cue Start
% Hz Attend, Col Attend, Training Cond
trig.cue = NaN(n.Hz_main,n.cols,n.attnstates);

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

trig.motion = NaN(n.Hz_main,n.cols,n.attnstates, n.dirs);

trig.motion(1,1,:,:) = [
    101 102 103 104;
    105 106 107 108
    ];

trig.motion(1,2,:,:) = trig.motion(1,1,:,:)+8;
trig.motion(2,1,:,:) = trig.motion(1,2,:,:)+8;
trig.motion(2,2,:,:) = trig.motion(2,1,:,:)+8;

% Feedback start
trig.feedback = 222;
