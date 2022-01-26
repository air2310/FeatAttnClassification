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
            ylim([48 85])
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
    [~, ~, CI, ~] = ttest(DAT(1,:)) % 61.7898   63.5177 | 51.1662   53.4874
    [~, ~, CI, ~] = ttest(DAT(2,:)) % 62.0946   63.9584 | 51.9324   54.0140
    [~, p, ~, stats] = ttest(DAT(1,:), DAT(2,:)) %p = 0.1736, t =  -1.3949 | p = 0.1018, t = -1.6896
    clear data
    data{1} = DAT(1,:);
    data{2} = DAT(2,:);
    data = data';
    i = rm_raincloud_series(data, linecolours(ii_train,:), ii_train, 'viridis');
end

% plot settings
% legend(sets.str.trainstrings)
xlim([40 75])
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
        ylim([48 100])
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

ylim([48 100])

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
            ylim([48 100])
        case 2
            ylim([49.6 60])
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
    datplot = squeeze(nanmean(ACCMEAN(:, end, :, :, :,ii_train) , 1));
    
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
xlim([0 5])
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
datplot = squeeze(nanmean(ACCMEAN(:, end, :, :, :,1) , 1));
LStyles = {'-' '--'};

H = figure;
hold on
ALL_Ms = [];
for SS = 1:30
    
    dat = squeeze(nanmean(datplot(: , SS, :),3));%squeeze(HZSTATEDAT(AA,ii,:, SS));
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
% min(ALL_Ms) =    79.8934   63.6126
% max(ALL_Ms) =    94.5752   89.2455
% mean(ALL_Ms) =  84.7161   72.2269

% [~,p,~,stats] = ttest(ALL_Ms(:,1), ALL_Ms(:,2))
% p =   0
%  tstat: 21.5630
%        df: 29
%        sd: 3.1724
[sum(maxi==1) sum(maxi==2) sum(maxi==3) sum(maxi==4)]
%  0     1     8    21
%  8    18     3     1

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

datplot = squeeze(nanmean(nanmean(ACCMEAN(:, :, 4, :, :, :) ,1),2));
datplot(:, 5, :) = squeeze(nanmean(nanmean(ACCMEAN(:, :, 1, :, 5, :) ,1),2));
datplot = datplot(:, [5 6 1 2 3 4], :);

% Get mean
M = squeeze(nanmean(datplot,1));
E = nan(sets.n.stimstates, sets.n.methods);
for ii_method = 1:sets.n.methods
    E(:,ii_method) = ws_bars(squeeze(datplot(:,ii_method,:)));
end

h = figure;
[~,hE] = barwitherr(E',M, 'LineWidth', 3);
set(hE, 'CapSize', 10, 'linewidth', 3)

set(gca, 'xticklabel',{ 'zscore' 'LR_L2' 'MLP' 'LDA' 'KNN' 'SVM'}, 'tickdir', 'out', 'LineWidth', 3)
box('off')

ylim([48 100])
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

datplot = squeeze(nanmean(nanmean(ACCMEAN(:, :, 4, :, :, :) ,1),2));
datplot(:, 5, :) = squeeze(nanmean(nanmean(ACCMEAN(:, :, 1, :, 5, :) ,1),2));
datplot = datplot(:, [5 6 1 2 3 4], :);

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


set(gca, 'yticklabel',{'SVM' 'KNN' 'LDA' 'MLP' 'LR' 'zscore'}, 'tickdir', 'out', 'LineWidth', 3)
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
datplot = squeeze(nanmean(nanmean(ACCMEAN(:, :, :, :, :, :) ,1),3));
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
anovan(ACC,{TRAINTYPE ALGORITHM WINDOW},3,3,varnames)

% partialetasquared = SS(effect)/ SS(effect) + SS(Error)
% for 3 way interaction
eta_p = 11830.5/(11830.5+22370.2) % = 0.3459
% 2 way traintype by window
eta_p = 21298.8/(21298.8+22370.2) % = 0.4877
% 2 way traintype by algorithm
eta_p = 41769.5/(41769.5+22370.2) % = 0.6512
% for sliding window
eta_p = 46493/(46493+22370.2) % = 0.6752
% for training type
eta_p = 46727.4/(46727.4+22370.2) % =  0.6763


% Overall effect of training type
datuse = squeeze(nanmean(nanmean(datplot,1), 3))
M = mean(datuse) % 62.8401   52.6500
SD = std(datuse) % 2.2924    2.7599


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
datplot = squeeze(nanmean(nanmean(nanmean(ACCMEAN(:, :, :, :, :, ii_train) ,1),3),5));
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
% ---------------------------------------------------------
%   WINDOW   10878.6     4    2719.65    334.8   3.8769e-72
%   Error     1177.9   145       8.12
%   Total    12056.5   149

% partialetasquared = SS(effect)/ SS(effect) + SS(Error)
eta_p = 10878.6/(10878.6+1177.9) % = 0.9023

% % Overall effect of window
M = mean(datplot') % 54.1077   55.5478   59.3866   68.6072   76.5515
SD = std(datplot') % 0.9913    1.6527    2.4602    3.6230    4.2099

% SingleFReq

% Multifreq
%   Source   Sum Sq.   d.f.   Mean Sq.    F       Prob>F
% ---------------------------------------------------------
%   WINDOW    420.03     4    105.008    9.25   1.09459e-06
%   Error    1645.61   145     11.349
%   Total    2065.64   149

% partialetasquared = SS(effect)/ SS(effect) + SS(Error)
eta_p = 420.03/(420.03+1645.61) % = 0.2033

% % Overall effect of window
M = mean(datplot') %  50.4992   51.3190   52.4079   53.9591   55.0649
SD = std(datplot') % 0.5957    1.5723    2.6576    4.2108    5.3967

%% Get average trajectory over chunk size increase

dat = squeeze(nanmean(nanmean(nanmean(nanmean(ACCMEAN(:, :, :, :, :, 2) ,1),3), 5),6));

figure; plot(sets.timing.secs.chunksizes , mean(dat'), 'x-')

%  fit with inverse exponential function
x = sets.timing.secs.chunksizes;
y =  mean(dat')'; % y(1) = [];

myfittype=fittype('a*(1-exp(-b*(x-c)))',...
    'dependent', {'y'}, 'independent',{'x'},'coefficients', {'a' 'b' 'c'});

myfit=fit(x',y,myfittype,'StartPoint',[80 0.5 0.5]);

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
ii_train = 1
datplot = squeeze(nanmean(nanmean(nanmean(ACCMEAN(:, :, :, :, :, ii_train) ,1),2),3))';
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
% Multifreq
%   Source      Sum Sq.   d.f.   Mean Sq.     F        Prob>F
% --------------------------------------------------------------
%   ALGORITHM   17181.4     5    3436.28    440.54   9.21293e-97
%   Error        1357.2   174       7.8
%   Total       18538.6   179
% partialetasquared = SS(effect)/ SS(effect) + SS(Error)
eta_p = 17181.4/(17181.4+1357.2) % = 0.9268

% % Overall effect of window
M = mean(datplot') %  57.4676   61.8795   67.8344   81.7302   52.1765   55.9526
SD = std(datplot') %  3.0250    3.5546    1.2379    2.2676    3.2194    2.8244

% SingleFreq

% Multifreq
%   Source      Sum Sq.   d.f.   Mean Sq.    F    Prob>F
% ------------------------------------------------------
%   ALGORITHM     65.14     5    13.0273    1.6   0.1621
%   Error       1415.31   174     8.134
%   Total       1480.45   179
%
% partialetasquared = SS(effect)/ SS(effect) + SS(Error)
eta_p = 65.14/(65.14+1415.31) % = 0.0440

% % Overall effect of window
M = mean(datplot') %   52.5908  83 53.8297   52.1121   52.8865   52.0223   52.4586
SD = std(datplot') % 2.6103    3.2682    2.3451    2.8236    3.1412    2.8230

% test against chance
[~,p, CI, stats] = ttest(mean(datplot), 50)
% p =1.2343e-05
% CI =51.6195   53.6806
%     tstat: 5.2592
%        df: 29
%        sd: 2.7599
mean(mean(datplot)) %52.6500

%% Get multifreq ordered performance
ii_train = 1
datplot = squeeze(nanmean(nanmean(nanmean(ACCMEAN(:, :, :, :, :, ii_train) ,1),2),3))';
order = [4 3 2 1 6 5]

% classifier
sets.str.methods{order}
%  'SVM' 'KNN' 'LDA' 'MLP' 'LR_L2' 'zscore'

% Means
mean(datplot(order, :)')
%    81.7302   67.8344   61.8795   57.4676   55.9526   52.1765

% Stats
for ii = 1:6
    disp(sets.str.methods{order(ii)})
    [~,p, CI, stats] = ttest(datplot(order(ii), :)',datplot(order(end), :)')
end
% p =    8.4508e-29    2.5818e-23    2.3101e-19   4.3807e-14    2.8185e-11
% 
% CI =
%    80.8834   67.3721   60.5522   56.3381   54.8979   
%    82.5769   68.2966   63.2068   58.5972   57.0072  
%    
%     tstat: [ 46.5282 29.8287 21.4938 13.5626 10.3791 ]

       

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
ii_train = 1
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
datplot = squeeze(nanmean(nanmean(nanmean(ACCMEAN(:, :, :, :, [ 1 2 3 4 6], :) ,1),2),5));
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


varnames = {'HZSTATE'};
[p,tbl,stats,terms] = anovan(ACC(1:120),{HZSTATE(1:120)},3,3,varnames)
varnames = {'HZSTATE'};
[p,tbl,stats,terms] = anovan(ACC(121:240),{HZSTATE(121:240)},3,3,varnames)

% report - no main effect or interaction with Hzstate - therefore no more
% follow up

% Partial Eta Squared!
% https://www.frontiersin.org/articles/10.3389/fpsyg.2013.00863/full

% Get all the sums of squares
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