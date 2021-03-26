%% Load EEG DATA
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
xlabel('Sample number')
ylabel('Trigger Amp.')
title('Triggers')

%% Correction for Subject 15
% an electrode became corrupted mid-experiment. This was replaced with the
% average of the remaining channels. 
if SUB == 15
    EEG(1811000:end,4) = mean(EEG(1811000:end,[1 2 3 5]),2);
end

%% replace noise for plotting

EEG2 = EEG;
EEG2(abs(EEG2)>70) = NaN; % For plotting purposes, we remove datapoints greater than 70 (i.e., blinks and such). These are left in during classification though. 

%% Observe EEG

figure; hold on
for CC = 1:size(EEG2,2)
    plot(EEG2(fs*5:end, CC)+200*CC)
end

set(gca, 'ytick', [])
xlabel('Sample #')
ylabel('EEG Channel')
title('Overview of EEG data')

%% Find triggers of interest
n.trials = length(find(ismember(TYPE, trig.trial(:,:,ATTNSTATE))));
LABELS_EEG = [];
idx_trials = [];
for HH = 1:n.Hz_main
    COND = trig.trial(HH,:,ATTNSTATE);
    idx_trials = [idx_trials; find(ismember(TYPE, COND))];
    
    LABELS_EEG = [LABELS_EEG; ones(length(find(ismember(TYPE, COND))),1).*HH];
end

%% Get trials
start = LATENCY(idx_trials)+lim.x_long(1);
stop = LATENCY(idx_trials)+lim.x_long(2);

TRIAL_EEG = NaN(n.x_long, n.chans, n.trials);
for TT = 1:n.trials
    
    tmp = EEG(start(TT):stop(TT),:);
    tmp = detrend(tmp, 'linear');
    tmp = tmp - tmp(1,:);
    
    TRIAL_EEG(:,:,TT) = tmp;
    
end

%% Trial settings
n.trials = length(LABELS_EEG);
n.channels = size(TRIAL_EEG,2);

%% Shuffle trial order
idx.shuffle = randperm(n.trials);
TRIAL_EEG = TRIAL_EEG(:,:,idx.shuffle);
LABELS_EEG = LABELS_EEG(idx.shuffle);

%% Erase noise for pictures

TRIAL_EEG2 = TRIAL_EEG;
TRIAL_EEG2(abs(TRIAL_EEG2)>70) = NaN;

TRIAL_EEG3 = TRIAL_EEG;
TRIAL_EEG3(abs(TRIAL_EEG2)>70) = 0;

%% Plot FFT spectrums

% run single trial FFT

tmp = abs( fft( TRIAL_EEG2 ) )/n.x;
tmp(2:end-1,:,:) = tmp(2:end-1,:,:)*2;

% calculate amp
AMP = NaN(n.x_long, 2);
AMP(:,1) = squeeze(nanmean(nanmean(tmp(:,:,LABELS_EEG==1),2),3));
AMP(:,2) = squeeze(nanmean(nanmean(tmp(:,:,LABELS_EEG==2),2),3));


h = figure; hold on;

plot(f_long, AMP(:,1));
plot(f_long, AMP(:,2));

xlim([2 20])
xlabel('Freq. (Hz)')
ylabel('FFT Amp (µV)')

legend({'Attend 6 Hz' 'Attend 7.5 Hz'})
tit = [str.sub ' ' str.attnstate{ATTNSTATE} ' FFT Spectrum'];
title(tit)

saveas(h, [direct.results tit '.png'])


%% Topographies 
[~,tmp1] = min(abs(f_long - Hz(1)));
[~,tmp2] = min(abs(f_long - Hz(2)));

idx.Hz_long = [tmp1 tmp2];

TOPO = NaN(n.Hz_main, n.channels, n.cols);
TOPO(:,:,1) = squeeze(nanmean(tmp(idx.Hz_long,:,LABELS_EEG==1),3));
TOPO(:,:,2) = squeeze(nanmean(tmp(idx.Hz_long,:,LABELS_EEG==2),3));


LIMIT = [min(TOPO(:)) max(TOPO(:))];
str.Hz = {[num2str(Hz(1)) ' Hz'] [num2str(Hz(2)) ' Hz']};

h = figure;
count = 0;
for HH = 1:n.Hz_main
    for HH_attd = 1:n.Hz_main
        count = count + 1;
        subplot(2,2,count)
        
        % update map
        map = makemap(TOPO(HH,:,HH_attd));
        
        % plot
        [C,h] = contourf(map);

        set(h,'LineColor','none')

        caxis(LIMIT)
        axis('off')
        colorbar
        
        title([str.Hz{HH} ' attend ' str.Hz{HH_attd} ])
    end
end
colormap(inferno)
tit = [str.sub ' ' str.attnstate{ATTNSTATE} ' TOPOS'];
suptitle(tit)

saveas(h, [direct.results tit '.png'])

%% Plot ERPs

ERP = NaN(n.x_long, 2);

ERP(:,1) = squeeze(nanmean(nanmean(TRIAL_EEG(:,:,LABELS_EEG==1),2),3));
ERP(:,2) = squeeze(nanmean(nanmean(TRIAL_EEG(:,:,LABELS_EEG==2),2),3));

h = figure; hold on;

plot(t_long, ERP(:,1));
plot(t_long, ERP(:,2));

legend({'Attend 6 Hz' 'Attend 7.5 Hz'})
xlabel('Time (s)')
ylabel('Amp (µV)')

tit = [str.sub ' ' str.attnstate{ATTNSTATE} ' ERPs'];
title(tit)

saveas(h, [direct.results tit '.png'])

%% for saving

datavis.AMP = AMP;
datavis.ERP = ERP;
datavis.TOPO = TOPO;
%% Pilot data
% load([direct.data 'RTDecodingPilotData.mat']) %pilot data, already formatted and preprocessed. 
