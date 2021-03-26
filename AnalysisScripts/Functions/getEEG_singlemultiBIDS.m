% Load EEG DATA
bids.taskname = 'FeatAttnDec';
filename.bids.EEG = [bids.substring '_task-' bids.taskname '_eeg'];

cfg            = [];
cfg.dataset    = [direct.data filename.bids.EEG '.eeg'];
cfg.continuous = 'yes';
cfg.channel    = 'all';
data           = ft_preprocessing(cfg);

% Assign data
EEG = data.trial{1}(1:n.chans,:)';
TRIG = data.trial{1}(n.chans+1,:)';

% Get Triggers
tmp = [0; diff(TRIG)];
LATENCY = find(tmp~=0);
TYPE = TRIG(LATENCY);

% Plot Triggers
h = figure;
hold on;
plot(TRIG)
stem(LATENCY, TYPE)

%% Correction for Subject 15
if SUB == 15
    EEG(1811000:end,4) = mean(EEG(1811000:end,[1 2 3 5]),2);
end

%% replace noise for plotting

EEG2 = EEG;
EEG2(abs(EEG2)>150) = NaN;

%% Find triggers of interest

for ATTNSTATE = 1:2
    
    n.trials = 80;%length(find(ismember(TYPE, trig.trial(:,:,ATTNSTATE))));
    LABELS_EEG.(str.attnstate{ATTNSTATE}) = [];
    idx_trials = [];
    for HH = 1:n.Hz_main
        COND = trig.trial(HH,:,ATTNSTATE);
        idx_trials = [idx_trials; find(ismember(TYPE, COND))];
        
        LABELS_EEG.(str.attnstate{ATTNSTATE}) = [LABELS_EEG.(str.attnstate{ATTNSTATE}); ones(length(find(ismember(TYPE, COND))),1).*HH];
    end
    
    %% Get trials
    start = LATENCY(idx_trials)+lim.x_long(1);
    stop = LATENCY(idx_trials)+lim.x_long(2);
    
    TRIAL_EEG.(str.attnstate{ATTNSTATE}) = NaN(n.x_long, n.chans, n.trials);
    for TT = 1:n.trials
        
        tmp = EEG(start(TT):stop(TT),:);
        tmp = detrend(tmp, 'linear');
        tmp = tmp - tmp(1,:);
        
        TRIAL_EEG.(str.attnstate{ATTNSTATE})(:,:,TT) = tmp;
        
    end
    
    %% Trial settings
%     n.trials = length(LABELS_EEG.(str.attnstate{ATTNSTATE}));
    n.channels = size(TRIAL_EEG.(str.attnstate{ATTNSTATE}),2);
    
    %% Shuffle trial order
    idx.shuffle = randperm(n.trials);
    TRIAL_EEG.(str.attnstate{ATTNSTATE}) = TRIAL_EEG.(str.attnstate{ATTNSTATE})(:,:,idx.shuffle);
    LABELS_EEG.(str.attnstate{ATTNSTATE}) = LABELS_EEG.(str.attnstate{ATTNSTATE})(idx.shuffle);
    
end
