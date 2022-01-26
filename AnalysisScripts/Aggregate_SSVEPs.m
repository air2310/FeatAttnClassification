%% Main Script for comparison of multiple methods of Real-time, singletrial, attentional selectivity calculation.
% Created by Angela I. Renton on 25/11/21. Email: angie.renton23@gmail.com

%% start with a clean slate
clear
clc
close all
addpath('Functions/');

% set random seed.
rng('default') % For reproducibility

%% Set decoding run options

% run individuals vs. collate group
runopts.individuals = 1;
runopts.collate = 1;

% Exclude motion epochs
runopts.excludemotepochs = 1; % 1 = exclude, 2 = include. 

%% Generic metadata
sets = setup_metadata();

%% Loop through subjects
if runopts.individuals
    % Preallocate
     n_electrodes = 5;
        n.x = 18000;
        ERP = NaN(n.x, n_electrodes, sets.n.Hz_f1, sets.n.cols, sets.n.traintypes, sets.n.sub_ids);
        FFT = NaN(n.x, n_electrodes, sets.n.Hz_f1, sets.n.cols, sets.n.traintypes, sets.n.sub_ids);
        SSVEP = NaN(sets.n.Hz_f1, n_electrodes, sets.n.Hz_f1, sets.n.cols, sets.n.traintypes, sets.n.sub_ids);

        %% Run
    for SUB = 1:sets.n.sub_ids
        %% Subject settings
        % Exclude excluded participants
        if ismember(SUB, [7 19])
            continue
        end
        
        % Subject strings and directoris.
        runopts.subject = SUB;
        disp(['Running subject :' num2str(runopts.subject)])
        sets = setup_subject_directories(sets, runopts);
        
        %% Load and organise EEG data
        disp('Loading EEG data')
        trialeeg = get_eeg(sets);
        
        %% Get colcond labels
        trialeeg.labels_colhzcond = (trialeeg.labels_colcued == trialeeg.labels_hzcued)+1; % 2 = black 6 Hz, 1 = black 7.5 Hz
        
        %% Get ERPs. 
        
        for ii_stimstate = 1:2
            for ii_colhzcond = 1:2
                for ii_hzattend = 1:2
                    % Get index
                    idx = trialeeg.labels_stimstate == ii_stimstate & trialeeg.labels_colhzcond == ii_colhzcond & trialeeg.labels_hzcued == ii_hzattend;
                    disp(sum(idx))
                    
                    % Get ERP
                    ERP(:,:,ii_hzattend, ii_colhzcond, ii_stimstate, SUB) = nanmean(trialeeg.dat(:,:,idx),3);
                    
                    % Fast Fourier Transform.
                    tmp = abs( fft( ERP(:,:,ii_hzattend, ii_colhzcond, ii_stimstate, SUB) ) )/n.x;
                    tmp(2:end-1,:) = tmp(2:end-1,:)*2;
                    FFT(:,:,ii_hzattend, ii_colhzcond, ii_stimstate, SUB) = tmp;
                    
                    % SSVEP
                    SSVEP(:,:,ii_hzattend, ii_colhzcond, ii_stimstate, SUB) = tmp(sets.Hz.epoch_long.f1_idx,:);
                end
            end
        end
    end
end

%% Plot!
for ii_stimstate = 1:2
    
    h = figure;
    for ii_colhzcond = 1:2;
        
        % Set colours
        switch ii_stimstate
            case 1
                colsuse = [121, 129, 187;
                    171, 179, 237]./255;
            case 2
                colsuse = [ 91 190 198;
                    141 240 248]./255;
        end
        
        % Get data
        subplot(1,2,ii_colhzcond);   hold on;
        
        DAT = squeeze(mean(nanmean(SSVEP(:,[1 2 3],:, ii_colhzcond, ii_stimstate, :), 2),4));
        DAT(:, :,[7 19]) = [];
        
        % DAT = DAT - mean(mean(DAT,2),1);% Subtract mean of frequency value.
        DAT = DAT - mean(DAT,2); % Subtract mean of frequency value.
        
        %     DAT = DAT - mean(DAT); % Need to subtract subject mean across hzstate
        %     conds?
        for ii_hzattend = 1:2
            clear data
            for ii = 1:2
                data{ii} = squeeze(DAT(ii,ii_hzattend,:));
            end
            data = data';
            
            % plot
            i = rm_raincloud_series(data, colsuse(ii_hzattend,:),  ii_hzattend, 'viridis');
        end
        
        % plot settings
        %     legend(sets.str.Hzattend)
        % xlim([45 75])
        set(gca, 'FontName', 'arial',  'LineWidth', 3, 'tickdir', 'out', 'box','off', 'Yticklabel', {'7.5' '6.0'} )
        ylabel('Frequency (Hz)')
        xlabel('Amplitude (µV)')
        
        title(sets.str.colcond{ii_colhzcond})
        
    end
    
    % save
    tit = ['SSVEP Amplitude by colour condition ' sets.str.testtrainopts{ii_stimstate}];
    suptitle(tit)
    saveas(h, [sets.direct.results_group tit '.png'])
    saveas(h, [sets.direct.results_group tit '.eps'], 'epsc')
    print(h,'-painters', '-depsc', [sets.direct.results_group  tit '.eps'])
    
end

%% ANOVA!

%% stats on Hzstate results
% 3 way anova
% data
ii_stimstate=2;
DAT = squeeze(nanmean(SSVEP(:,[1 2 3],:, :, ii_stimstate, :), 2));
DAT(:, :,:,[7 19]) = [];

ACC = [];

% factors
HZ = cell(sets.n.subs*2*2*2,1);
HZATTEND = cell(sets.n.subs*2*2*2,1);
COLCOND = cell(sets.n.subs*2*2*2,1);

% create data and variable vectors
idx = 1:30;
for HH = 1:2
    for HH_at = 1:2
        for CC= 1:2
            %             dat = squeeze(HZSTATEDAT(AA, ii, HH, :));
            dat = squeeze(DAT(HH, HH_at,CC,:));
            
            % data
            ACC = [ACC; dat];
            
            % factors
            HZ(idx) =  {sets.str.Hz{HH}};
            HZATTEND(idx) =  {sets.str.Hz{HH_at}};
            COLCOND(idx) =  {sets.str.colcond{CC}};
            
            
            idx = idx+30;
        end
    end
end
varnames = {'HZ';'HZATTEND'; 'COLCOND'};
[p,tbl,stats,terms] = anovan(ACC,{HZ HZATTEND COLCOND},3,3,varnames)

% report - no main effect or interaction with Hzstate - therefore no more
% follow up

% Partial Eta Squared!
% https://www.frontiersin.org/articles/10.3389/fpsyg.2013.00863/full

% Get all the sums of squares
tbl{5,1} = 'HZ_HZATTEND'
tbl{6,1} = 'HZ_COLCOND'
tbl{7,1} = 'HZATTEND_COLCOND'
tbl{8,1} =  'HZ_HZATTEND_COLCOND'
for N = 2:9 
    SSS.(tbl{N,1}) = tbl{N,2}
end

% Get all the sums of squares
for N = 2:9
    partialetasquared.(tbl{N,1}) = SSS.(tbl{N,1}) / (SSS.(tbl{N,1}) + SSS.Error)
end

% Get data
datuse = squeeze(mean(DAT, 3));
mean(datuse, 3) 

% 1.4679    1.3340
% 0.9304    1.0500

% 6 Hz   1.3004    0.1315
% 5.5 Hz 0.1256    0.9013


[~,~,CI,~]=ttest(squeeze(datuse(1,:,:))')
[~,~,CI,~]=ttest(squeeze(datuse(2,:,:))')

[~,p,~,stats]=ttest(squeeze(datuse(1,1,:))', squeeze(datuse(1,2,:))')
[~,p,~,stats]=ttest(squeeze(datuse(2,1,:))', squeeze(datuse(2,2,:))')

% CI 61.4893   63.5386   65.3032   66.1557
%    63.6863   65.4175   66.6997   67.4924
% tstat: [23.4362 31.5192 46.8705 51.4850]
%        df: [29 29 29 29]
%        sd: [2.9419 2.5159 1.8699 1.7898]

%% Attended - Unattended bar plot. 
ii_stimstate = 2;
DAT = squeeze(nanmean(SSVEP(:,:,:, :, ii_stimstate,:), 4));
DAT(:, :,:,[7 19]) = [];

switcher = [ 1 2; 2 1];
ssvep_attend = NaN(size(DAT));
for ii_hzattend = 1:2
   ssvep_attend(:,:,ii_hzattend,:) = DAT(switcher(:, ii_hzattend), :,ii_hzattend, :);
end
ssvep_attend = squeeze(mean(ssvep_attend,3));
ssvep_affect = squeeze(ssvep_attend(1,:,:) - ssvep_attend(2,:,:));


clear data
for ii = 1:5
    data{ii} = ssvep_affect(ii,:);
end
data = data';

% Set colours
switch ii_stimstate
    case 1
        colsuse = [146, 149, 212]./255;
    case 2
        colsuse = [ 110 210 210]./255;
end

h = figure;
i = rm_raincloud(data, colsuse, 'viridis');

% plot settings

set(gca, 'FontName', 'arial',  'LineWidth', 3, 'tickdir', 'out', 'box','off', 'Yticklabel', {'O2' 'O1' 'POz' 'Oz' 'Iz'} )
ylabel('Electrode')
xlabel('Cued - Uncued Amplitude (µV)')
line([0 0], [-1 10],'LineWidth', 3, 'Color', [0 0 0], 'LineStyle', '--')

tit = ['SSVEP Amplitude by electrode ' sets.str.testtrainopts{ii_stimstate}];
title(tit)
saveas(h, [sets.direct.results_group tit '.png'])
print(h,'-painters', '-depsc', [sets.direct.results_group  tit '.eps'])