%% start with a clean slate
clear
clc
close all
addpath('Functions/');

% set random seed.
rng('default') % For reproducibility

%% Generic metadata
sets = setup_metadata();

%% Includemotepochs

runopts.excludemotepochs = 1; % 1 = exclude, 2 = include.
str_mot = sets.str.excludemotepochs{runopts.excludemotepochs};

%% Plotting settings

direct.toolbox = 'Functions/PlottingToolboxes/';
direct.violin = [direct.toolbox 'RainCloudPlots-master/tutorial_matlab/'];
direct.colormaps = [direct.toolbox 'Colormaps/'];
direct.barwitherr = [direct.toolbox 'barwitherr/'];
addpath(direct.violin)
addpath(direct.colormaps)
addpath(direct.barwitherr)
tmp = viridis;
linecolours = tmp( round(linspace(1,length(tmp),sets.n.methods)),:);

%% Load data

ACCMEAN = NaN(sets.n.cols, sets.n.chunksizes, sets.n.hzstates, sets.n.subs, sets.n.methods, sets.n.traintypes);
for ii_train = 1:sets.n.traintypes
    for ii_method = 1:sets.n.methods
        % load data
        decodestring = sets.str.methods{ii_method};
        load([sets.direct.results_group 'ACCURACY_' decodestring '_' sets.str.trainstrings{ii_train} str_mot '.mat'], 'ACCMEAN_ALL');
        ACCMEAN(:,:,:,:,ii_method, ii_train) = ACCMEAN_ALL;
    end
end

%% Compare effect of colour counterbalancing.
for ii_train = 1:sets.n.traintypes
    
    h = figure;
    
    % get data
    datplot = squeeze(nanmean(nanmean(ACCMEAN(:,:,:,:,:,ii_train), 2),3));
    M = squeeze(nanmean(datplot,2));
    E = nan(sets.n.cols, sets.n.methods);
    for ii_col = 1:sets.n.cols
        E(ii_col,:) = ws_bars(squeeze(datplot(ii_col,:,:)));
    end
    
    % Plot
    [hh, ee] = barwitherr(E',M');
    set(gca,'XTickLabel',sets.str.methods)
    set(hh(1),'FaceColor','k','LineWidth', 2);
    set(hh(2),'FaceColor','w','LineWidth', 2);
    set(ee(1),'LineWidth', 2);
    set(ee(2),'LineWidth', 2);
    
    % limits
    switch ii_train
        case 1
            ylim([49 60])
        case 2
            ylim([49 60])
    end
    line([0 7], [50 50],'LineWidth', 3, 'Color', [1 0 0], 'LineStyle','--')
    set(gca, 'FontName', 'arial',  'LineWidth', 3, 'tickdir', 'out', 'box','off')
    
    
    % labels
    xlabel('Algorithm')
    ylabel('Accuracy (%)')
    legend(sets.str.colcond, 'location', 'NorthWest')
    
    tit = ['Accuracy by colour condition ' sets.str.trainstrings{ii_train} str_mot];
    suptitle(tit)
    saveas(h, [sets.direct.results_group tit '.png'])
    saveas(h, [sets.direct.results_group tit '.eps'], 'epsc')
end
return
%% Compare effect of colour counterbalancing - raincloud plot.

h = figure; hold on;

% cond 1
for ii_train = 1:sets.n.traintypes
    DAT = squeeze(nanmean(nanmean(nanmean(ACCMEAN(:,:,:,:,:,ii_train), 2),3),5));
    [~, ~, CI, ~] = ttest(DAT(1,:)) % 51.9449   54.4883 | 51.1371   53.4881 
    [~, ~, CI, ~] = ttest(DAT(2,:)) % 52.3066   55.0202 | 51.9596   54.0993 
    [~, p, ~, stats] = ttest(DAT(1,:), DAT(2,:)) %p = 0.2608, t = -1.1468 | p = 0.0757, t(29) =  -1.8423 | 
    clear data
    data{1} = DAT(1,:);
    data{2} = DAT(2,:);
    data = data';
    i = rm_raincloud_series(data, linecolours(ii_train,:), ii_train, 'viridis');
end

% plot settings
% legend(sets.str.trainstrings)
xlim([45 70])
set(gca, 'FontName', 'arial',  'LineWidth', 3, 'tickdir', 'out', 'box','off', 'Yticklabel', sets.str.colcond)
ylabel('Flicker Condition')
xlabel('Accuracy (%)')

% save
tit = ['Mean Accuracy by colour condition ' str_mot];
suptitle(tit)
saveas(h, [sets.direct.results_group tit '.png'])
saveas(h, [sets.direct.results_group tit '.eps'], 'epsc')
print(h,'-painters', '-depsc', [sets.direct.results_group  tit '.eps'])


% Multifreq
% p = 0.1736
%     tstat: -1.3949
%        df: 29
%        sd: 1.4638
% M = 62.6537   63.0265
% STD = 2.3137    2.4957

% Singlefreq
% p = 0.1018
%     tstat: -1.6896
%        df: 29
%        sd: 2.0955
% M = 52.3268   52.9732
% STD = 3.1081    2.7874


%% Compare effect of Distractor presence - raincloud plot

h = figure;

% Pre-plot data sorting
DAT = squeeze(nanmean(nanmean(nanmean(nanmean(ACCMEAN(:,:,:,:,:,:), 2),3),5),1));
[~,idx_ombre] = sort(DAT(:,1)); % sort just the first column
sorteddat = DAT(idx_ombre, :)'; % sort the whole matrix using the sort indices

clear data
data{1} = sorteddat(1,:);
data{2} = sorteddat(2,:);
data = data';
i = rm_raincloud(data, linecolours(ii_train,:), 'viridis');


% plot settings
% legend(sets.str.trainstrings)
xlim([45 75])
set(gca, 'FontName', 'arial',  'LineWidth', 3, 'tickdir', 'out', 'box','off', 'Yticklabel', {'Present' 'Absent'} )
ylabel('Distractor')
xlabel('Accuracy (%)')

% save
tit = ['Mean Accuracy by Distractor condition ' str_mot];
suptitle(tit)
saveas(h, [sets.direct.results_group tit '.png'])
saveas(h, [sets.direct.results_group tit '.eps'], 'epsc')
print(h,'-painters', '-depsc', [sets.direct.results_group  tit '.eps'])


%% Plot results together

for ii_train = 1:sets.n.traintypes
    
    h = figure;
    for ii_hzstate = 1:sets.n.hzstates
        subplot(2,2,ii_hzstate)
        
        % get data
        datplot = squeeze(nanmean(ACCMEAN(:,:,ii_hzstate, :, :,ii_train) , 1));
        M = squeeze(nanmean(datplot,2));
        E = nan(sets.n.chunksizes, sets.n.methods);
        for ii_method = 1:sets.n.methods
            E(:,ii_method) = ws_bars(datplot(:,:,ii_method)');
        end
        
        % Plot
        hold on;
        for ii_method = 1:sets.n.methods
            errorbar(sets.timing.secs.chunksizes,M(:,ii_method),E(:,ii_method),'linewidth', 2,'color', linecolours(ii_method,:)')
        end
        
        % limits
        ylim([48 70])
        line([0 4], [50 50],'LineWidth', 3, 'Color', [0 0 0])
        set(gca, 'FontName', 'arial',  'LineWidth', 3, 'tickdir', 'out')
        set(gca, 'FontName', 'arial',  'LineWidth', 3)
        
        % labels
        xlabel('Sliding window size')
        ylabel('Accuracy (%)')
        legend(sets.str.methods, 'location', 'NorthWest')
        title(sets.str.HzState{ii_hzstate})
        
    end
    tit = ['Accuracy detailed ' sets.str.trainstrings{ii_train} str_mot];
    suptitle(tit)
    saveas(h, [sets.direct.results_group tit '.png'])
    saveas(h, [sets.direct.results_group tit '.eps'], 'epsc')
end
return

%% Effect of chunksize
h = figure;
for ii_train = 1:sets.n.traintypes
    
    % Get best frequency for each subject
    datplot_tmp = squeeze(nanmean(ACCMEAN(:,:,:, :, :,ii_train) , 1));
    datplot = NaN(sets.n.chunksizes, sets.n.subs, sets.n.methods);
    for ii_method = 1:sets.n.methods
        for ii_sub = 1:sets.n.subs
            [tmp, tmp_idx] = max(squeeze(mean(datplot_tmp(:,:,ii_sub,ii_method),1)));
            datplot(:,ii_sub, ii_method) = datplot_tmp(:,tmp_idx, ii_sub, ii_method);
        end
    end
    
    % Get mean
    M = squeeze(nanmean(datplot,2));
    E = nan(sets.n.chunksizes, sets.n.methods);
    for ii_method = 1:sets.n.methods
        E(:,ii_method) = ws_bars(datplot(:,:,ii_method)');
    end
    
    % Plot
    hold on;
    for ii_method = 1:sets.n.methods
        switch ii_train
            case 1
                errorbar(sets.timing.secs.chunksizes,M(:,ii_method),E(:,ii_method),'linewidth', 2,'color', linecolours(ii_method,:)')
            case 2
                errorbar(sets.timing.secs.chunksizes,M(:,ii_method),E(:,ii_method),'linewidth', 2,'color', linecolours(ii_method,:)', 'LineStyle', ':')
        end
        
    end
    
    
end
% limits

ylim([48 70])

line([0 4], [50 50],'LineWidth', 3, 'Color', [0 0 0])
set(gca, 'FontName', 'arial',  'LineWidth', 3, 'tickdir', 'out')
set(gca, 'FontName', 'arial',  'LineWidth', 3)

% labels
xlabel('Sliding window size')
ylabel('Accuracy (%)')
legend(sets.str.methods, 'location', 'NorthWest')

% Title
tit = ['Accuracy by chunksize combined ' str_mot];
suptitle(tit)
saveas(h, [sets.direct.results_group tit '.png'])
saveas(h, [sets.direct.results_group tit '.eps'], 'epsc')


%% Effect of chunksize

for ii_train = 1:sets.n.traintypes
    
    % Get best frequency for each subject
    datplot_tmp = squeeze(nanmean(ACCMEAN(:,:,:, :, :,ii_train) , 1));
    datplot = NaN(sets.n.chunksizes, sets.n.subs, sets.n.methods);
    for ii_method = 1:sets.n.methods
        for ii_sub = 1:sets.n.subs
            [tmp, tmp_idx] = max(squeeze(mean(datplot_tmp(:,:,ii_sub,ii_method),1)));
            datplot(:,ii_sub, ii_method) = datplot_tmp(:,tmp_idx, ii_sub, ii_method);
        end
    end
    
    % Get mean
    M = squeeze(nanmean(datplot,2));
    E = nan(sets.n.chunksizes, sets.n.methods);
    for ii_method = 1:sets.n.methods
        E(:,ii_method) = ws_bars(datplot(:,:,ii_method)');
    end
    
    h = figure;
    % Plot
    hold on;
    for ii_method = 1:sets.n.methods
        errorbar(sets.timing.secs.chunksizes,M(:,ii_method),E(:,ii_method),'linewidth', 2,'color', linecolours(ii_method,:)')
    end
    
    % limits
    switch ii_train
        case 1
            ylim([48 70])
        case 2
            ylim([48 70])
    end
    
    
    line([0 4], [50 50],'LineWidth', 3, 'Color', [0 0 0])
    set(gca, 'FontName', 'arial',  'LineWidth', 3, 'tickdir', 'out')
    set(gca, 'FontName', 'arial',  'LineWidth', 3)
    
    % labels
    xlabel('Sliding window size')
    ylabel('Accuracy (%)')
    legend(sets.str.methods, 'location', 'NorthWest')
    
    % Title
    tit = ['Accuracy by chunksize ' sets.str.trainstrings{ii_train} str_mot];
    suptitle(tit)
    saveas(h, [sets.direct.results_group tit '.png'])
    saveas(h, [sets.direct.results_group tit '.eps'], 'epsc')
    
end

%% effect of Hz features
str.HzstateMetricsMixed = {'MLP - Multi' 'LDA - Multi'    'KNN - Multi' 'SVM - Multi' 'LR - Multi'  'MLP - SingleToMult' 'LDA - SingleToMulti'  'KNN - SingleToMult'  'SVM - SingleToMult' 'LR - SingleToMult'};

h = figure; hold on;
for ii_train = 1:sets.n.traintypes
    % Get data to plot
    datplot = squeeze(nanmean(nanmean(ACCMEAN(:, [3 4 5], :, :, :,ii_train) , 1),2));
    
    % Get mean
    M = squeeze(nanmean(datplot,2));
    E = nan(sets.n.hzstates, sets.n.methods);
    for ii_method = 1:sets.n.methods
        E(:,ii_method) = ws_bars(datplot(:,:,ii_method)');
    end
    
    % Plot
    methodsuse = [1 2 3 4 6];
    for ii = 1:5
        ii_method = methodsuse(ii);
        switch ii_train
            case 1
                errorbar(1:4,M(:,ii_method),E(:,ii_method),'linewidth', 2,'color', linecolours(ii_method,:)', 'linestyle','--')
            case 2
                errorbar(1:4,M(:,ii_method),E(:,ii_method),'linewidth', 2,'color', linecolours(ii_method,:)', 'linestyle','-')
        end
    end
end

set(gca, 'xtick', 1:4,  'xticklabel', sets.str.HzState,'tickdir', 'out')
xlim([0.5 4.5])
ylim([50 62])
set(gca, 'FontName', 'arial',  'LineWidth', 3)
box('off')

legend(str.HzstateMetricsMixed, 'location', 'NorthWest')

xlabel('Frequency Features Used')
ylabel('Variance across Trials')

tit = ['Frequency contributions MIXED' str_mot];
title(tit)
saveas(h, [sets.direct.results_group tit  '.png'])
saveas(h, [sets.direct.results_group tit  '.eps'], 'epsc')


%% Hzstatedat individual differences
ii_train = 1;
datplot = squeeze(nanmean(ACCMEAN(:, end, :, :, 2,1) , 1));
LStyles = {'-' '--'};

H = figure;
hold on
ALL_Ms = [];
for SS = 1:30
    
    dat = squeeze(datplot(: , SS));%squeeze(HZSTATEDAT(AA,ii,:, SS));
    [M1, maxi(SS)] = max(dat);
    M2 = dat(4);%min(dat);
    M = [ M1 M2];
    E = zeros(2,1);
    errorbar(M,E', 'linewidth', 2, 'color', linecolours(1,:), 'CapSize', 12, 'LineStyle', LStyles{ii_train});
    
    ALL_Ms = [ALL_Ms; M];
end


xlim([0 3])
% ylim([50 57]);
% set(gca, 'xtick', 1:4,  'xticklabel', str.HzState,'tickdir', 'out')

set(gca, 'FontName', 'arial',  'LineWidth', 3)
box('off')

% range
% min(ALL_Ms) =     52.3371   49.1200
% max(ALL_Ms) =    90.8908   85.1612
% mean(ALL_Ms) = 66.7995   60.2456
% std(ALL_Ms)= 9.5182    8.1361

% [~,p,~,stats] = ttest(ALL_Ms(:,1), ALL_Ms(:,2))
% p =   2.4217e-11
%     tstat: 10.4476
%        df: 29
%        sd: 3.4359
[sum(maxi==1) sum(maxi==2) sum(maxi==3) sum(maxi==4)]
% 15    14     0     1

%% Find timepoints that are significantly greater than chance

datplot = squeeze(nanmean(nanmean(nanmean(nanmean(ACCMEAN(:, :, :, :, :, 1) ,1),3),5),6));
M = mean(datplot,2);

% - calculate 95% confidince intervals
E = NaN(size(M));
P = NaN(size(M));
for tt = 1:sets.n.chunksizes
    [~,P(tt),CI,~] = ttest(datplot(tt,:), 50);
    E(tt) = diff(CI)./2;
end

h = figure;
[~,hE] = barwitherr(E,M, 'linewidth', 3);
set(hE, 'CapSize', 20, 'linewidth', 3)
set(gca, 'xticklabel',sets.timing.secs.chunksizes, 'tickdir', 'out', 'LineWidth', 3)
box('off')

ylim([48 85])
colormap([0.9 0.9 0.9])
line([0 6], [50 50], 'color', 'r', 'linewidth', 3)

xlabel('Chunk Size')
ylabel('Accuracy (%)')

tit = ['Chunksize Effect MIXED' str_mot];
title(tit)
saveas(h, [sets.direct.results_group tit  '.png'])
saveas(h, [sets.direct.results_group tit  '.eps'], 'epsc')


%% Plot group result
chunksizesuse = [3:5];
datplot = squeeze(nanmean(nanmean(nanmean(ACCMEAN(:, chunksizesuse, [1 2], :, :, :) ,1),2),3));
datplot(:, 5, :) = squeeze(nanmean(nanmean(ACCMEAN(:, :, 1, :, 5, :) ,1),2));
datplot = datplot(:, [5 6 3 1  4 2], :);

% Get mean
M = squeeze(nanmean(datplot,1));
E = nan(sets.n.stimstates, sets.n.methods);
for ii_method = 1:sets.n.methods
    E(:,ii_method) = ws_bars(squeeze(datplot(:,ii_method,:)));
end

h = figure;
[~,hE] = barwitherr(E',M, 'LineWidth', 3);
set(hE, 'CapSize', 10, 'linewidth', 3)

set(gca, 'xticklabel',{ 'zscore' 'KNN' 'MLP' 'LR_L2'  'SVM' 'LDA'  }, 'tickdir', 'out', 'LineWidth', 3)
box('off')

ylim([49 70])
legend(sets.str.trainstrings, 'location', 'NorthWest')
colormap([1 1 1; 0 0 0])
line([0 7], [50 50], 'color', 'r', 'linewidth', 3)

ylabel('Chunk Size')
xlabel('Accuracy')


tit = ['Algorithm effect MIXED' str_mot];
title(tit)
saveas(h, [sets.direct.results_group tit  '.png'])
saveas(h, [sets.direct.results_group tit  '.eps'], 'epsc')


%% Plot group result - raincloud
chunksizesuse = [3:5];
datplot = squeeze(nanmean(nanmean(nanmean(ACCMEAN(:, chunksizesuse, [1 2], :, :, :) ,1),2),3));
datplot(:, 5, :) = squeeze(nanmean(nanmean(ACCMEAN(:, :, 1, :, 5, :) ,1),2));
datplot = datplot(:, [5 6 3 1  4 2], :);

% Create figure
h = figure; hold on;

% Pre-plot data sorting
for ii_train = 1:sets.n.traintypes
    DAT = datplot(:,:, ii_train)';
    clear data
    for ii = 1:6
        data{ii} = DAT(ii,:);
    end
    data = data';
    
    i = rm_raincloud_series(data, linecolours(ii_train,:),  ii_train, 'viridis');
end


set(gca, 'yticklabel',{'LDA' 'SVM'   'MLP' 'KNN' 'LR'  'zscore'}, 'tickdir', 'out', 'LineWidth', 3)
box('off')


legend(sets.str.trainstrings, 'location', 'NorthWest')
line( [50 50],[-0.7 5.5], 'color', 'k', 'linewidth', 3, 'lineStyle', '--')
xlim([45 90])
ylim([-0.7 5.5])
ylabel('Algorithm')
xlabel('Accuracy (%)')


tit = ['Algorithm effect MIXED raincloud' str_mot];
title(tit)
saveas(h, [sets.direct.results_group tit  '.png'])
saveas(h, [sets.direct.results_group tit  '.eps'], 'epsc')
print(h,'-painters', '-depsc', [sets.direct.results_group  tit '.eps'])
%% Stats
%% stats on big interaction
% 3 way anova
% data
datplot = squeeze(nanmean(nanmean(ACCMEAN(:, :, [1 2], :, :, :) ,1),3));
ACC = [];

% factors
TRAINTYPE = cell(sets.n.subs*2*6*5,1);
ALGORITHM = cell(sets.n.subs*2*6*5,1);
WINDOW = cell(sets.n.subs*2*6*5,1);

% create data and variable vectors
n.multistates = 2;
idx = 1:30;
ii_methods = [ 1 2 3 4 5 6];
sets.str.chunksizes = {'.25' '.5' '1' '2' '4'};
for AA = 1:n.multistates
    for ii = 1:sets.n.methods
        for WW = 1:sets.n.chunksizes
            %             dat = squeeze(HZSTATEDAT(AA, ii, HH, :));
            dat = squeeze(datplot(WW, :,ii_methods(ii), AA))';
            
            % data
            ACC = [ACC; dat];
            
            % factors
            TRAINTYPE(idx)  = {sets.str.testtrainopts{AA}};
            ALGORITHM(idx) = {sets.str.methods{ii_methods(ii)}};
            WINDOW(idx) ={sets.str.chunksizes{WW}};
            
            idx = idx+30;
        end
    end
end

varnames = {'TRAINTYPE';'ALGORITHM';'WINDOW'};
[p,tbl,stats,terms] = anovan(ACC,{TRAINTYPE ALGORITHM WINDOW},3,3,varnames)

% Partial Eta Squared!
% https://www.frontiersin.org/articles/10.3389/fpsyg.2013.00863/full

% Get all the sums of squares
tbl{5,1} = 'TRAINTYPE_ALGORITHM'
tbl{6,1} = 'TRAINTYPE_WINDOW'
tbl{7,1} = 'ALGORTHM_WINDOW'
tbl{8,1} = 'TRAINTYPE_ALGORITHM_WINDOW'
for N = 2:9
    SSS.(tbl{N,1}) = tbl{N,2}
end

% Get all the sums of squares
for N = 2:9
    partialetasquared.(tbl{N,1}) = SSS.(tbl{N,1}) / (SSS.(tbl{N,1}) + SSS.Error)
end

% Overall effect of training type
datuse = squeeze(nanmean(nanmean(datplot,1), 3))
M = mean(datuse) % 54.0740   52.9148
SD = std(datuse) % 3.6125    2.9084


%% Power calcs
%  {'MLP'}    {'LDA'}    {'KNN'}    {'SVM'}    {'zscore'}    {'LR_L2'}
datplot = squeeze(nanmean(nanmean(nanmean(ACCMEAN(:, :, :, :, [5 6], 2) ,1),2),3));
TRAINTYPE = cell(sets.n.subs*2,1);
idx = 1:30;
ACC = [];
for WW = 1:2
    dat = squeeze(datplot(:,WW));
    
    % data
    ACC = [ACC; dat];
    
    % factors
    TRAINTYPE(idx)  = {sets.str.testtrainopts{WW}};
    idx = idx+30;
end
varnames = {'TRAINTYPE'};
anovan(ACC,{TRAINTYPE },3,3,varnames)

mean(datplot)

x = datplot

[rows, cols] = size(x);

sub_mean = nanmean(x, 2); %disp( sub_mean );
grand_mean = nanmean( sub_mean, 1 );

x2 = x - ( repmat( sub_mean, 1, cols ) - grand_mean );

ws_std = mean(nanstd( x2' ),2);


ws_std = sqrt(213.89);
n_trials = 80
bs_std = sqrt(mean(std(x))^2 - ws_std^2/n_trials)

%% stats on sliding window size
% 3 way anova
% data
ii_train = 2
datplot = squeeze(nanmean(nanmean(nanmean(ACCMEAN(:, :, [1 2], :, :, ii_train) ,1),3),5));
ACC = [];

% factors
WINDOW = cell(sets.n.subs*5,1);

% create data and variable vectors

idx = 1:30;
sets.str.chunksizes = {'.25' '.5' '1' '2' '4'};

for WW = 1:sets.n.chunksizes
    % data
    dat = squeeze(datplot(WW, :))';
    ACC = [ACC; dat];
    
    % factors
    WINDOW(idx) ={sets.str.chunksizes{WW}};
    idx = idx+30;
end

% Run ANOVA
varnames = {'WINDOW'};
anovan(ACC,{WINDOW},3,3,varnames)

% Multifreq
%   Source   Sum Sq.   d.f.   Mean Sq.     F       Prob>F   
% ----------------------------------------------------------
%   WINDOW   1725.19     4    431.297    20.76   1.52589e-13
%   Error    3013.07   145     20.78                        
%   Total    4738.26   149                                  

% partialetasquared = SS(effect)/ SS(effect) + SS(Error)
eta_p = 1725.19/(4738.26) % = 0.9023

% % Overall effect of window
M = mean(datplot') % 54.1077   55.5478   59.3866   68.6072   76.5515
SD = std(datplot') % 0.9913    1.6527    2.4602    3.6230    4.2099

% SingleFReq

% Multifreq
%   Source   Sum Sq.   d.f.   Mean Sq.    F       Prob>F   
% ---------------------------------------------------------
%   WINDOW    474.46     4    118.615    9.49   7.67428e-07
%   Error    1812.76   145     12.502                      
%   Total    2287.22   149                                 

% partialetasquared = SS(effect)/ SS(effect) + SS(Error)
eta_p = 474.46/(2287.22) % = 0.2033

% % Overall effect of window
M = mean(datplot') %  50.4992   51.3190   52.4079   53.9591   55.0649
SD = std(datplot') % 0.5957    1.5723    2.6576    4.2108    5.3967

%% Get average trajectory over chunk size increase
ii_train=2
dat = squeeze(nanmean(nanmean(nanmean(nanmean(ACCMEAN(:, :, [1 2], :, :, ii_train) ,1),3), 5),6));

figure; plot(sets.timing.secs.chunksizes , mean(dat'), 'x-')

%  fit with inverse exponential function
x = sets.timing.secs.chunksizes;
y =  mean(dat')'; % y(1) = [];

[~,~,CI,~] = ttest(dat')
y=CI(1,:)'
myfittype=fittype('a*(1-exp(-b*(x-c)))',...
    'dependent', {'y'}, 'independent',{'x'},'coefficients', {'a' 'b' 'c'});

myfit=fit(x',y,myfittype,'StartPoint',[50 0.5 0.5]);

% Asymptote
myfit.a

% Find index at 99% of the asymptote
x2 = 0.25:0.25:40;
y2 = myfit.a*(1-exp(-myfit.b*(x2-myfit.c)));
[~,iH] = min(abs(y2 - max(y2)*0.99))

% Data length at 99% of the asymtote:
x2(iH)

% Plot results
h= figure; hold on;
x2 = 0.25:0.25:4;
plot(x2, myfit.a*(1-exp(-myfit.b*(x2-myfit.c))), 'g--')
plot(x,y, 'g-x')


%% stats on Algorithm
% 3 way anova
% data
ii_train = 2
% datplot = squeeze(nanmean(nanmean(nanmean(ACCMEAN(:, :, [1 2], :, :, ii_train) ,1),2),3))';
datplot = squeeze(nanmean(nanmean(nanmean(ACCMEAN(:, [3 4 5], [1 2], :, :, ii_train) ,1),2),3))';

ACC = [];

% factors
ALGORITHM = cell(sets.n.subs*6,1);

% create data and variable vectors

idx = 1:30;

for WW = 1:sets.n.methods
    % data
    dat = squeeze(datplot(WW, :))';
    ACC = [ACC; dat];
    
    % factors
    ALGORITHM(idx) ={sets.str.methods{WW}};
    idx = idx+30;
end

% Run ANOVA
varnames = {'ALGORITHM'};
anovan(ACC,{ALGORITHM},3,3,varnames)
return

% partialetasquared = SS(effect)/ SS(effect) + SS(Error)
eta_p = 914.64/(6568.96) % = 0.139

% % Overall effect of algorithm
% sets.str.methods =   {'MLP'}    {'LDA'}    {'KNN'}    {'SVM'}    {'zscore'}    {'LR_L2'}
M = mean(datplot') %  56.2349   60.5793   55.5472   56.8406   53.1512   55.1102
SD = std(datplot') %    5.7614    7.0975    5.1304    5.7245    4.7186    5.4821

% SingleFreq
% 
%   Source      Sum Sq.   d.f.   Mean Sq.    F     Prob>F
% -------------------------------------------------------
%   ALGORITHM    120.57     5    24.1141    1.25   0.2862
%   Error       3346.67   174    19.2338                 
%   Total       3467.24   179                                                   
%
% partialetasquared = SS(effect)/ SS(effect) + SS(Error)
eta_p = 120.57/(3467.24) % = 0.0348

% % Overall effect of window
M = mean(datplot') %    52.8708   53.9904   52.7746   53.1834   51.9938   52.6758
SD = std(datplot') % 2.7432    3.3598    2.8855    2.8467    3.1132    3.0154

% test against chance
[~,p, CI, stats] = ttest(mean(datplot), 50)
% p =9.5211e-06
% CI =52.5640   55.7351
%     tstat: 5.3526
%        df: 29
%        sd: 4.2462
mean(mean(datplot)) %54.1496

%% Get multifreq ordered performance
ii_train = 1
datplot = squeeze(nanmean(nanmean(nanmean(ACCMEAN(:, [3 4 5], [1 2], :, :, ii_train) ,1),2),3))';
order = [4 3 2 1 6 5]

% classifier
sets.str.methods{order}
%  'SVM' 'KNN' 'LDA' 'MLP' 'LR_L2' 'zscore'

% Means
mean(datplot(order, :)')
%     56.8406   55.5472   60.5793   56.2349   55.1102   53.1512

% Stats
for ii = 1:6
    disp(sets.str.methods{order(ii)})
    [~,p, ~, stats] = ttest(datplot(order(ii), :)',datplot(order(end), :)')
    [~,~, CI, ~] = ttest(datplot(order(ii), :)')
end
% p =    3.1755e-05  0.0010  6.0804e-09  2.3927e-04   0.0046
%     tstat: [ 4.9192 3.6533  8.1094   4.1885   3.0733 ]



       

%% SVM at low sliding window sizes
ii_train = 1
% SVM
datplot = squeeze(nanmean(nanmean(nanmean(ACCMEAN(:, 1:3, :, :, 4, ii_train) ,1),2),3))';

mean(datplot) % 74.6243
[~,p, CI, stats] = ttest(datplot) % CI 73.4187   75.8300

% Remaining classifiers
datplot = squeeze(nanmean(nanmean(nanmean(ACCMEAN(:, 1:3, :, :, [1 2 3 5 6], ii_train) ,1),2),3))';

mean(mean(datplot)) % 52.6920
[~,p, CI, stats] = ttest(mean(datplot)) % CI 52.1647   53.2192

%% Prepare max accuacy table
ii_train = 2
order = [5 2 6 4 1 3]
sets.str.methods{order}
dat = squeeze(nanmean(ACCMEAN(:, end, :, :, order, ii_train) ,1));

besthzstate = NaN(sets.n.subs, sets.n.methods)
maxval = NaN(sets.n.subs, sets.n.methods)
for sub = 1:sets.n.subs
   [maxval(sub,:), besthzstate(sub,:)] = max(squeeze(dat(:,sub,:)))
end


% KNN best performance
mean(maxval(:,end)) %99.0630
[~,p, CI, stats] = ttest(maxval(:,end)) % CI98.8551 99.2709

%% stats on Hzstate results
% 3 way anova
% data
datplot = squeeze(nanmean(nanmean(ACCMEAN(:, :, :, :, :, :) ,1),2));
ACC = [];

% factors
TRAINTYPE = cell(sets.n.subs*2*5*4,1);
ALGORITHM = cell(sets.n.subs*2*5*4,1);
HZSTATE = cell(sets.n.subs*2*5*4,1);

% create data and variable vectors
n.multistates = 2;
idx = 1:30;
ii_methods = [ 1 2 3 4 6];
for AA = 1:n.multistates
    for ii = 1:sets.n.methods-1
        for HH = 1:sets.n.hzstates
            %             dat = squeeze(HZSTATEDAT(AA, ii, HH, :));
            dat = squeeze(datplot(HH, :,ii_methods(ii), AA))';
            
            % data
            ACC = [ACC; dat];
            
            % factors
            TRAINTYPE(idx)  = {sets.str.testtrainopts{AA}};
            ALGORITHM(idx) = {sets.str.methods{ii_methods(ii)}};
            HZSTATE(idx) ={sets.str.HzState{HH}};
            
            idx = idx+30;
        end
    end
end

varnames = {'TRAINTYPE';'ALGORITHM';'HZSTATE'};
anovan(ACC,{TRAINTYPE ALGORITHM HZSTATE},3,3,varnames)

% report - no main effect or interaction with Hzstate - therefore no more
% follow up

%% stats on Hzstate results
% 3 way anova
% data
datplot = squeeze(nanmean(nanmean(nanmean(ACCMEAN(:, [3 4 5], :, :, [ 1 2 3 4 6], :) ,1),2),5));
ACC = [];

% factors
TRAINTYPE = cell(sets.n.subs*2*4,1);
HZSTATE = cell(sets.n.subs*2*4,1);

% create data and variable vectors
n.multistates = 2;
idx = 1:30;
for AA = 1:n.multistates
    for HH = 1:sets.n.hzstates
        %             dat = squeeze(HZSTATEDAT(AA, ii, HH, :));
        dat = squeeze(datplot(HH, :,AA))';
        
        % data
        ACC = [ACC; dat];
        
        % factors
        TRAINTYPE(idx)  = {sets.str.testtrainopts{AA}};
        HZSTATE(idx) ={sets.str.HzState{HH}};
        
        idx = idx+30;
    end
end

varnames = {'TRAINTYPE';'HZSTATE'};
[p,tbl,stats,terms] = anovan(ACC,{TRAINTYPE HZSTATE},3,3,varnames)

% 
% varnames = {'HZSTATE'};
% [p,tbl,stats,terms] = anovan(ACC(1:120),{HZSTATE(1:120)},3,3,varnames)
% varnames = {'HZSTATE'};
% [p,tbl,stats,terms] = anovan(ACC(121:240),{HZSTATE(121:240)},3,3,varnames)

% report - no main effect or interaction with Hzstate - therefore no more
% follow up

% Partial Eta Squared!
% https://www.frontiersin.org/articles/10.3389/fpsyg.2013.00863/full

% Get all the sums of squares
clear SSS partialetasquared
tbl{4,1} = 'TRAINTYPE_HZSTATE'
for N = 2:5 
    SSS.(tbl{N,1}) = tbl{N,2}
end

% Get all the sums of squares
for N = 2:4
    partialetasquared.(tbl{N,1}) = SSS.(tbl{N,1}) / (SSS.(tbl{N,1}) + SSS.Error)
end

% Get data
datplot = squeeze(nanmean(nanmean(nanmean(ACCMEAN(:, :, :, :, [ 1 2 3 4 6], 1) ,1),2),5));
mean(datplot') %62.5878   64.4781   66.0015   66.8241
[~,p,CI,stats]=ttest(datplot',50)
% CI 61.4893   63.5386   65.3032   66.1557
%    63.6863   65.4175   66.6997   67.4924
% tstat: [23.4362 31.5192 46.8705 51.4850]
%        df: [29 29 29 29]
%        sd: [2.9419 2.5159 1.8699 1.7898]