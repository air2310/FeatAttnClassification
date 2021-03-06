clear
clc
% close all

%% Directories


direct.toolbox = '..\..\..\toolboxes\';
addpath(direct.toolbox)
addpath([direct.toolbox 'Colormaps\'])
addpath(genpath([direct.toolbox 'kakearney-boundedline-pkg-50f7e4b\']))
addpath([direct.toolbox 'barwitherr\'])
addpath([direct.toolbox 'topoplot\'])


direct.data = '..\data\EEG_mat\';
direct.results = ['..\results\' ];

%% Setup Settings
options.collate=1;
setupSettingsBIDS

%% To Do
% - picture of SSVEP amps for window sizes/ 
str.HzstateMetrics = {'LDA' 'KNN' 'MLP'};
str.traintype = {'multi' 'singletomulti'};

%% organise subjects

subsuse = [1:6 8:18 20:32];
% subsuse = [1 2 3 4 5 6 8:9];

n.subs = length(subsuse);

%% preallocate
for ATTNSTATE = 1%:2
    n.chunks_max = 60;
    n.trials = 80;
    n.train_freq_cond = 4;
    
    ACC_ARRANGED.CCA = NaN(n.chunks_max, n.trials, n.subs, n.chunksizes);
    ACC_ARRANGED.zscore = NaN(n.chunks_max, n.trials, n.subs, n.chunksizes);
    ACC_ARRANGED.LDA = NaN(n.chunks_max, n.trials, n.subs, n.chunksizes, n.train_freq_cond );
    ACC_ARRANGED.KNN = NaN(n.chunks_max, n.trials, n.subs, n.chunksizes, n.train_freq_cond );
    ACC_ARRANGED.MLP = NaN(n.chunks_max, n.trials, n.subs, n.chunksizes, n.train_freq_cond );
    
    DATAVIS.AMP = NaN(n.x_long,2,n.subs);
    DATAVIS.ERP = NaN(n.x_long,2,n.subs);
    DATAVIS.TOPO = NaN(2,5,2,n.subs);
    
    %% get data
    for SS = 1:n.subs
        %% directory
        
        SUB = subsuse(SS);
        str.sub = ['S' num2str(SUB)];
        
        direct.resultsSUB = ['..\results\' str.sub '\'];
        direct.resultsSUB = ['U:\FeatAttnClassification\Results\' str.sub '\'];
        
        %% load data
        
        tic
%         if ATTNSTATE == 3
%             load([direct.resultsSUB str.sub str.attnstate{ATTNSTATE}  'results.mat'], 'ACCURACY_arranged')
%             tmp = ACCURACY_arranged.CCA;
%             
%             load([direct.resultsSUB str.sub str.attnstate{ATTNSTATE}  'results COLOUR.mat'], 'ACCURACY_ALL', 'ACCURACY_arranged', 'dat', 'datavis')
%             ACCURACY_arranged.CCA = tmp;
%         else
            load([direct.resultsSUB str.sub str.attnstate{ATTNSTATE}  'results.mat'], 'ACCURACY_ALL', 'ACCURACY_arranged', 'dat', 'datavis')
%         end
        
        
        load([direct.resultsSUB str.sub 'BehaveResults.mat'], 'BEHAVE')
        toc
        
        
        %% Allocate behaviour
        
        BEHAVEALL.ACC_By_Attn(:,:,SS) = BEHAVE.ACC_By_Attn;
        BEHAVEALL.ACC_By_Col(:,:,SS) = BEHAVE.ACC_By_Col;
        BEHAVEALL.RT_By_AttnandCol(:,:,SS) = BEHAVE.RT_By_AttnandCol;
        
        %% Allocate accuracy
        for chunksizeiterator = 1:n.chunksizes
            % get chunk size
            chunksize = samples.chunksizes(chunksizeiterator); % Choose chunksize
            samplepoints =  chunksize  : samples.slidingwindow : samples.trial; % The points we would hypothetically be calculating this at in realtime
            n.chunks = length(samplepoints);
            
            % Allocate
            ACC_ARRANGED.CCA(n.chunks_max-n.chunks+1:end,:,SS,chunksizeiterator) = ACCURACY_arranged.CCA{chunksizeiterator}(:,1:n.trials);
            ACC_ARRANGED.zscore(n.chunks_max-n.chunks+1:end,:,SS,chunksizeiterator) = ACCURACY_arranged.zscore{chunksizeiterator}(:,1:n.trials);
            
            for HZSTATE = 1:n.train_freq_cond
                ACC_ARRANGED.LDA(n.chunks_max-n.chunks+1:end,:,SS,chunksizeiterator,HZSTATE) = ACCURACY_arranged.LDA.(str.HzState{HZSTATE}){chunksizeiterator}(:,1:n.trials);
                ACC_ARRANGED.KNN(n.chunks_max-n.chunks+1:end,:,SS,chunksizeiterator,HZSTATE) = ACCURACY_arranged.KNN.(str.HzState{HZSTATE}){chunksizeiterator}(:,1:n.trials);
                ACC_ARRANGED.MLP(n.chunks_max-n.chunks+1:end,:,SS,chunksizeiterator,HZSTATE) = ACCURACY_arranged.MLP.(str.HzState{HZSTATE}){chunksizeiterator}(:,1:n.trials);
            end
        end
        
        %% allocate data visualisation
        if ATTNSTATE <3
            DATAVIS.AMP(:,:,SS) = datavis.AMP;
            DATAVIS.ERP(:,:,SS) = datavis.ERP;
            DATAVIS.TOPO(:,:,:,SS) = datavis.TOPO;
        end
        
    end
        
  
    %% Plot ERPs
    if ATTNSTATE <3
        clear M E
        M = mean(DATAVIS.ERP,3);
        
        for cc = 1:2
            E(:,cc) = ws_bars(squeeze(DATAVIS.ERP(:,cc,:))');
        end
        
        h = figure;
        boundedline(t_long,M,E,'cmap', cool(4))
        
        xlabel('Time (s)')
        ylabel('EEG amp (?V)')
        legend(str.Hzattend);
        
        tit = ['Grand Average ERP ' str.attnstate{ATTNSTATE}];
        title(tit);
        saveas(h, [direct.results tit '.png'])
    end
    
    %% Plot FFT spectrum
    if ATTNSTATE <3
        clear E
        M = mean(DATAVIS.AMP,3);
        
        for cc = 1:2
            E(:,cc) = ws_bars(squeeze(DATAVIS.AMP(:,cc,:))');
        end
        COLS = [93 97 194;
            92 232 205]./255;
        h = figure;
        boundedline(f_long,M,E,'cmap', COLS)
        xlim([2 20])
        ylim([0 12])
        xlabel('Frequency (Hz)')
        ylabel('FFT amp (?V)')
        legend(str.Hzattend);
        
        set(gca, 'FontName', 'arial',  'LineWidth', 3, 'tickdir', 'out')
         
        tit = ['Grand Average FFT spectrum ' str.attnstate{ATTNSTATE}];
        title(tit);
        set(gcf,'renderer','painters') 
        saveas(h, [direct.results tit '.png'])
        saveas(gcf, [direct.results tit '.eps'], 'epsc')
    end
    
%%     Plot AMP Bar 
    if ATTNSTATE <3
        
        tmp = DATAVIS.AMP(idx.Hz_long,:,:);
        % reshuffle tmp to attended/unattended
        attstates = [1 2; 2 1];
        for CC = 1:2          
           tmp(:,CC,:) = tmp(attstates(CC,:),CC,:);
        end
        dat = squeeze(mean(tmp,2));
        
        M = mean(dat,2);
        SSD = std(dat');
        
        E = ws_bars(dat')';

        h = figure;
        [~,hE] = barwitherr(E,M, 'linewidth', 3, 'FaceColor','k');
        set(hE, 'CapSize', 20, 'linewidth', 3)
%         colormap([0 0 0])
        xlim([0 3])
        switch ATTNSTATE
            case 1
                ylim([0 10])
            case 2
                ylim([0 10])
        end
        xlabel('Frequency')
        ylabel('FFT amp (?V)')
        
        box('off')
        set(gca, 'xticklabel', {'Attended' 'Unattended'}, 'FontName', 'arial',  'LineWidth', 3, 'tickdir', 'out')
         
        tit = ['Grand Average SSVEP Bar ' str.attnstate{ATTNSTATE}];
        title(tit);
        
        [~,p,~,stats] = ttest(dat(1,:), dat(2,:));
        
        text(1, 8.5, ['t(' num2str(stats.df) ') = ' num2str(stats.tstat) ', p = ' num2str(p)])
        saveas(h, [direct.results tit '.png'])
        saveas(h, [direct.results tit '.eps'], 'epsc')
    end
     
    %% Plot Topos 3
     load('chanlocs_5chan.mat', 'chanlocs')
    if ATTNSTATE <3
        tmp = mean(DATAVIS.TOPO,4);
        
        attstates = [1 2; 2 1];
        for CC = 1:2          
           tmp(:,:,CC) = tmp(attstates(CC,:),:,CC);
        end
        TOPO = mean(tmp,3);
        TOPO = [TOPO; TOPO(1,:) - TOPO(2,:)];
        
        % get limits
        LIMIT = NaN(2,3);
        tmp = TOPO(1:2,:);
        LIMIT(:,1) = [3 6];%[5 9.5]; %[min(tmp(:)) max(tmp(:))];
        LIMIT(:,2) = [3 6];%[5 9.5]; %[min(tmp(:)) max(tmp(:))];
        LIMIT(:,3) = [-6 6];%[-max(abs((TOPO(3,:)))) max(abs((TOPO(3,:))))];
        
        str.attention = {'Attended' 'Unattended' 'Attended - Unattended'};
       
        for ATTN = 1:3
            h = figure;
            
            % update map
            TOPO(ATTN,6:10) = min(LIMIT(:,ATTN));
%              TOPO(ATTN,6:10) = min(TOPO(ATTN,:));
            if ATTN == 2 && ATTNSTATE ==2;  TOPO(ATTN,6:10) = min(TOPO(ATTN,:)); end
            
            % plot
            topoplot(TOPO(ATTN,:), chanlocs, 'maplimits', LIMIT(:, ATTN), 'colormap', inferno, 'shading', 'interp', 'emarker', {'.','k',10,5}, 'style', 'map' );

            colorbar
%             colormap(inferno)
            colormap(viridis)
            if ATTN == 3; colormap(viridis); end
            
            tit = [str.attention{ATTN} ' '    str.attnstate{ATTNSTATE} ' TOPOS'];
            title(tit)
            saveas(h, [direct.results tit '.png'])
            saveas(h, [direct.results tit '.eps'], 'epsc')
        end

    end
    

    
    %% get feature based attn bar graph
    
    [~,tmp1] = min(abs(f_long - Hz(1)));
    [~,tmp2] = min(abs(f_long - Hz(2)));
    idx.tmp = [tmp1 tmp2];
    
    dat = DATAVIS.AMP(idx.tmp,:,:);
    
    attd = cat(2,squeeze(dat(1,1,:)), squeeze(dat(2,2,:)));
    unattd = cat(2,squeeze(dat(1,2,:)), squeeze(dat(1,2,:)));
    
    SSVEPs = [mean(attd,2) mean(unattd,2)];
    
    % ############# PLOT bar hereHERE ######################
    
    %% correlate
    if ATTNSTATE <3
     ACCDAT = squeeze(mean(mean(nanmean(ACC_ARRANGED.LDA(:,:,:,end,:),1),2),5)).*100;;
     
     h = figure;
     subplot(1,2,1); hold on;
     
     dat2 = SSVEPs(:,1)-SSVEPs(:,2);
     dat1 = ACCDAT;
     scatter(dat2,dat1,'k',  'linewidth', 3 )
     
     Fit = polyfit(dat2,dat1,1);
     plot(dat2, Fit(1)*dat2 + Fit(2), 'r', 'linewidth', 3)
     
     [r, p] = corr(dat1, dat2);
     switch ATTNSTATE
         case 1
            text(-2,65, ['r = ' num2str(r) ', p = ' num2str(p) ]) 
         case 2
            text(1,90, ['r = ' num2str(r) ', p = ' num2str(p) ])
     end
     
     xlabel('Attd. - UnAttd SSVEP (??V)')
     ylabel('Accuracy (%)')
     
     title('Feat based. Attn Vs. Acc')
     set(gca, 'FontName', 'arial',  'LineWidth', 3, 'tickdir', 'out')
     
     
%      subplot(1,3,2); hold on;
%      
%      dat2 = mean(SSVEPs,2);
%      dat1 = ACCDAT;
%      scatter(dat2,dat1, 'k', 'linewidth', 3 )
%      
%      Fit = polyfit(dat2,dat1,1);
%      plot(dat2, Fit(1)*dat2 + Fit(2), 'r', 'linewidth', 3)
%      
%      [r, p] = corr(dat1, dat2);
%      switch ATTNSTATE
%          case 1
%             text(7,65, ['r = ' num2str(r) ', p = ' num2str(p) ]) 
%          case 2
%              text(4.5,90, ['r = ' num2str(r) ', p = ' num2str(p) ])
%      end
%      
%     
%      
%      xlabel('SSVEP Amp (?V)')
%      ylabel('Accuracy (%)')
%      
%      title('SSVEP Vs. Acc')
%      set(gca, 'FontName', 'arial',  'LineWidth', 3, 'tickdir', 'out')
%      
     featattn = SSVEPs(:,1)-SSVEPs(:,2);
     behave = squeeze(BEHAVEALL.ACC_By_Attn(ATTNSTATE,1,:));
    behave = squeeze(BEHAVEALL.RT_By_AttnandCol(1,ATTNSTATE,:));
     subplot(1,2,2)
     hold on;
     
     dat2 =  featattn;
     dat1 =   behave ;
     scatter(dat2,dat1, 'k', 'linewidth', 3 )
     
     Fit = polyfit(dat2,dat1,1);
     plot(dat2, Fit(1)*dat2 + Fit(2), 'r', 'linewidth', 3)
     
     [r, p] = corr(dat1, dat2);
     switch ATTNSTATE
         case 1
             text(-2,80, ['r = ' num2str(r) ', p = ' num2str(p) ])
         case 2
             text(1,90, ['r = ' num2str(r) ', p = ' num2str(p) ])
     end
     
     xlabel('Attd. - UnAttd SSVEP (??V)')
     ylabel('Behave Accuracy (%)')
     set(gca, 'FontName', 'arial',  'LineWidth', 3, 'tickdir', 'out')
     title('SSVEP vs. behave acc')
     
     tit = [str.attnstate{ATTNSTATE} ' SSVEP Scatter plots'];
     suptitle(tit)
     
     set(gcf,'renderer','painters') 
     saveas(h, [direct.results tit '.png'])
     saveas(gcf, [direct.results tit '.eps'], 'epsc')
     
    end
  
    %% Behaviour
    
    if ATTNSTATE <3
        str.behave = {'correct' 'incorrect' 'miss' 'falsealarm'};
        behave = squeeze(BEHAVEALL.ACC_By_Attn(ATTNSTATE,:,:));
        M = mean(behave');
        STD = std(behave');
        E = ws_bars(behave');
        
        h = figure;
        
        [~,hE] = barwitherr(E,M, 'linewidth', 3);
        set(hE, 'CapSize', 20, 'linewidth', 3)
        
        set(gca, 'xticklabel', str.behave, 'tickdir', 'out', 'LineWidth', 3)
        box('off')
        ylabel('Responses (%)')
        xlabel('Response type')
        
        colormap([0 0 0])
        tit =  [str.attnstate{ATTNSTATE} ' Behaviour'];
        suptitle(tit)
        
        set(gcf,'renderer','painters')
        saveas(h, [direct.results tit '.png'])
        saveas(gcf, [direct.results tit '.eps'], 'epsc')
    end
    
% single freq - correct, incorrect, miss, false alarm
%     M =  87.7417    2.3250    6.0500    3.8833
%    STD =   14.0721    3.6579    7.0739    6.4999


% multifreq
% M =   62.1667    8.1667   26.0833    3.5833
% STD = 16.4672    8.5615   13.9652    6.4187


%% Reaction times

dat = squeeze(BEHAVEALL.RT_By_AttnandCol(1,:,:));% get data for attention split, not colour split

M = mean(dat');
% mult, single =  0.6438    0.5750
STD = std(dat');
% mult, single =  0.0758    0.0640

    %% line colours
    
    
    tmp = viridis;
    linecolours = tmp( round(linspace(1,length(tmp),5)),:);
    
    
    %% Summarise affect of additional info
    % ACC_ARRANGED.LDA(n.chunks_max-n.chunks+1:end,:,SS,chunksizeiterator,HZSTATE)
    
    str.HzstateMetrics = {'LDA' 'KNN' 'MLP'};
    
    % tmp = parula;
    % linecolours = tmp( round(linspace(20,length(tmp)-10,3)),:);
    
    h = figure; hold on
    
    for ii = 1:3
        DAT = squeeze(mean(mean(nanmean(ACC_ARRANGED.(str.HzstateMetrics{ii}),1),2),4)).*100;
        M = mean(DAT,1);
        E = ws_bars(DAT);
        
        errorbar(M,E, 'linewidth', 2, 'color', linecolours(ii+2,:))
        
    end
    
    xlim([0 5])
    if ATTNSTATE == 2
        ylim([50 100])
    else
        ylim([48 60])
        line([0 5], [50 50], 'color', 'k')
    end
    
    set(gca, 'xtick', 1:4,  'xticklabel', str.HzState,'tickdir', 'out')
    
    legend(str.HzstateMetrics)
    
    xlabel('Frequency Features Used')
    ylabel('Accuracy (%)')
    
    set(gca, 'FontName', 'arial',  'LineWidth', 3, 'tickdir', 'out')
     
    tit = ['Frequency contributions ' str.attnstate{ATTNSTATE}];
    title(tit)
    saveas(h, [direct.results tit  '.png'])
    saveas(h, [direct.results tit '.eps'], 'epsc') 
  
    %% Summarise affect of additional info - Trial by trial variance.
    % ACC_ARRANGED.LDA(n.chunks_max-n.chunks+1:end,:,SS,chunksizeiterator,HZSTATE)
    
    
    % tmp = parula;
    % linecolours = tmp( round(linspace(20,length(tmp)-10,3)),:);
    
    h = figure; hold on
    
    for ii = 1:3
        DAT = squeeze(var(squeeze(mean(nanmean(ACC_ARRANGED.(str.HzstateMetrics{ii}),1),4))));
        M = mean(DAT,1);
        E = ws_bars(DAT);
        
        errorbar(M,E, 'linewidth', 2, 'color', linecolours(ii+2,:))
        
    end
    
    xlim([0 5])
    ylim([ 0 0.03])
    set(gca, 'xtick', 1:4,  'xticklabel', str.HzState,'tickdir', 'out')
    
    legend(str.HzstateMetrics)
    
    xlabel('Frequency Features Used')
    ylabel('Variance across Trials')
    
    tit = ['Frequency contributions variance ' str.attnstate{ATTNSTATE}];
    title(tit)
    saveas(h, [direct.results tit  '.png'])
    
    %% Summarise effect of chunksize
    % ACC_ARRANGED.LDA(n.chunks_max-n.chunks+1:end,:,SS,chunksizeiterator,HZSTATE)
    % ACC_ARRANGED.CCA(n.chunks_max-n.chunks+1:end,:,SS,chunksizeiterator)
    
    str.algorithm = {'CCA' 'zscore' 'LDA' 'KNN' 'MLP'};
    
    % tmp = parula;
    % linecolours = tmp( round(linspace(5,length(tmp),5)),:);
    
    h = figure; hold on
    
    for ii = 1:5
        if ii <3
            DAT = squeeze(mean(nanmean(ACC_ARRANGED.(str.algorithm{ii}),1),2)).*100;
        else
            
            % Get optimal set of frequencies for each person
            tmp = squeeze(mean(mean(nanmean(ACC_ARRANGED.(str.algorithm{ii}),1),2),4));
            [tmp2,Hzstateuse] = max(tmp,[],2);

            % get data using optimum
            DAT = NaN(n.subs,n.chunksizes);
            for SS = 1:n.subs
                DAT(SS,:) =  squeeze(mean(nanmean(ACC_ARRANGED.(str.algorithm{ii})(:,:,SS,:,Hzstateuse(SS)),1),2)).*100;
%                 DAT(SS,:) =  squeeze(mean(nanmean(ACC_ARRANGED.(str.algorithm{ii})(:,:,SS,:,1),1),2)).*100;
            end
            
        end
        
        M = nanmean(DAT,1);
        E = ws_bars(DAT);
        
        errorbar( time.chunksizes, M,E, 'linewidth', 2, 'color', linecolours(ii,:), 'CapSize', 12)
        
    end
    
    xlim([0 4.25])
    
    if ATTNSTATE == 2
        ylim([45 100])
    else
        ylim([45 65])
    end
    
    
    legend(str.algorithm, 'location', 'NorthWest')
    line([0 7], [50 50],'LineWidth', 3, 'Color', [0 0 0])
%     set(gca, 'xtick',  time.chunksizes , 'xticklabels', time.chunksizes, 'tickdir', 'out')
 set(gca, 'FontName', 'arial',  'LineWidth', 3, 'tickdir', 'out')
    set(gca, 'FontName', 'arial',  'LineWidth', 3)
    
    xlabel('Chunk Size')
    ylabel('Accuracy (%)')
    
    tit = ['Algorithm by Chunk Size ' str.attnstate{ATTNSTATE}];
    title(tit)
    saveas(h, [direct.results tit '.png'])
    saveas(h, [direct.results tit '.eps'], 'epsc') 
    
    %% Summarise effect of chunksize - variance
    % ACC_ARRANGED.LDA(n.chunks_max-n.chunks+1:end,:,SS,chunksizeiterator,HZSTATE)
    % ACC_ARRANGED.CCA(n.chunks_max-n.chunks+1:end,:,SS,chunksizeiterator)
    
    str.algorithm = {'CCA' 'zscore' 'LDA' 'KNN' 'MLP'};
    
    % tmp = parula;
    % linecolours = tmp( round(linspace(5,length(tmp),5)),:);
    
    h = figure; hold on
    
    for ii = 1:5
        if ii <3
            DAT = squeeze(var(squeeze(nanmean(ACC_ARRANGED.(str.algorithm{ii}),1)),1));
        else
            tmp = squeeze(mean(mean(nanmean(ACC_ARRANGED.(str.algorithm{ii}),1),2),4));
            [tmp2,Hzstateuse] = max(tmp,[],2);
            
            % get data using optimum
            DAT = NaN(n.subs,n.chunksizes);
            for SS = 1:n.subs
                DAT(SS,:) =  squeeze(var(squeeze(nanmean(ACC_ARRANGED.(str.algorithm{ii})(:,:,SS,:,Hzstateuse(SS)),1)),1));
            end
            
        end
        
        M = mean(DAT,1);
        E = ws_bars(DAT);
        
        errorbar(time.chunksizes, M,E, 'linewidth', 2, 'color', linecolours(ii,:))
        
    end
    
    xlim([0 4.25])
    ylim([0 0.13])
    legend(str.algorithm, 'location', 'southEast')
    set(gca, 'xtick', 1:6, 'xticklabels', time.chunksizes, 'tickdir', 'out')
    
    
    xlabel('Chunk Size')
    ylabel('Variance across Trials')
    
    tit = ['Algorithm by Chunk Size variance' str.attnstate{ATTNSTATE}];
    title(tit)
    saveas(h, [direct.results tit '.png'])
    saveas(h, [direct.results tit '.eps'], 'epsc') 
    %% Summarise effect of chunksize
    % ACC_ARRANGED.LDA(n.chunks_max-n.chunks+1:end,:,SS,chunksizeiterator,HZSTATE)
    % ACC_ARRANGED.CCA(n.chunks_max-n.chunks+1:end,:,SS,chunksizeiterator)
%     for sub = 1:8
    str.algorithm = {'CCA' 'zscore' 'LDA' 'KNN' 'MLP'};
    
    % tmp = parula;
    % linecolours = tmp( round(linspace(5,length(tmp),5)),:);
    
    h = figure; hold on
    
    for ii = 1:5
        if ii <3
            DAT = squeeze(mean(nanmean(ACC_ARRANGED.(str.algorithm{ii})(:,:,:,2:end),4),2)).*100;
        else
            
            % Get optimal set of frequencies for each person
            tmp = squeeze(mean(mean(nanmean(ACC_ARRANGED.(str.algorithm{ii}),1),2),4));
            [tmp2,Hzstateuse] = max(tmp,[],2);
            
            % get data using optimum
            DAT = NaN(60,n.subs);
            for SS = 1:n.subs
                DAT(:,SS) =   squeeze(mean(mean(nanmean(ACC_ARRANGED.(str.algorithm{ii})(:,:,SS,2:end,Hzstateuse(SS)),4),2),5)).*100;
            end
            
        end
        
        M = mean(DAT,2);
        E = ws_bars(DAT');
        t =  (1  : samples.slidingwindow : samples.trial)./fs;
        
        plot(t, smooth(M), 'linewidth', 2, 'color', linecolours(ii,:))
        
    end
    
    %     xlim([0 7])
    
    if ATTNSTATE == 2
        ylim([45 100])
    else
        ylim([49 60])
        
    end
    line([0 15], [50 50], 'color', 'k')
    
    legend(str.algorithm, 'location', 'southEast')
    set(gca, 'FontName', 'arial',  'LineWidth', 3, 'tickdir', 'out')
    
    
    xlabel('Time in Trial (s)')
    ylabel('Accuracy (%)')
%     end
    tit = ['Algorithm by Time in trial ' str.attnstate{ATTNSTATE}];
    title(tit)
    saveas(h, [direct.results tit '.png'])
    saveas(h, [direct.results tit '.eps'], 'epsc') 
end


%% Combined data/plots
attnstatesuse = [1 3];% multifreq, single to multi.
n.chunks_max = 60;
n.trials = 80;
n.train_freq_cond = 4;
n.HzstateMetrics = 3;
n.Hzstates = 4;
n.attnstates_multi = 2;
    
% preallocate
HZSTATEDAT = NaN(n.attnstates_multi, n.HzstateMetrics, n.Hzstates, n.subs);
HZSTATEDAT_M = NaN(n.attnstates_multi, n.HzstateMetrics, n.Hzstates);
HZSTATEDAT_E = NaN(n.attnstates_multi, n.HzstateMetrics, n.Hzstates);

ACC_CHUNK = NaN(n.attnstates_multi, n.chunksizes, n.methods, n.subs);
 str.algorithm = {'CCA' 'zscore' 'LDA' 'KNN' 'MLP'};
% chunksizeEDAT = NaN(n.attnstates_multi, n.HzstateMetrics, n.Hzstates, n.subs);
    
    
for AA = 1:2
    
    ATTNSTATE = attnstatesuse(AA);

    ACC_ARRANGED.CCA = NaN(n.chunks_max, n.trials, n.subs, n.chunksizes);
    ACC_ARRANGED.zscore = NaN(n.chunks_max, n.trials, n.subs, n.chunksizes);
    ACC_ARRANGED.LDA = NaN(n.chunks_max, n.trials, n.subs, n.chunksizes, n.train_freq_cond );
    ACC_ARRANGED.KNN = NaN(n.chunks_max, n.trials, n.subs, n.chunksizes, n.train_freq_cond );
    ACC_ARRANGED.MLP = NaN(n.chunks_max, n.trials, n.subs, n.chunksizes, n.train_freq_cond );

    
    %% get data
    for SS = 1:n.subs
        %% directory
        
        SUB = subsuse(SS);
        str.sub = ['S' num2str(SUB)];
        direct.resultsSUB = ['U:\FeatAttnClassification\Results\' str.sub '\']
%         direct.resultsSUB = ['..\results\' str.sub '\'];
%         direct.resultsSUB = ['Z:\angela_renton\RTAttnselectMethods\RTAttnSelectMethods_SOLC\results\' str.sub '\'];
        %% load data
        
        load([direct.resultsSUB str.sub str.attnstate{ATTNSTATE}  'results.mat'], 'ACCURACY_ALL', 'ACCURACY_arranged', 'dat' )
        load([direct.resultsSUB str.sub 'BehaveResults.mat'], 'BEHAVE')
        
        %% Allocate behaviour
        
        BEHAVEALL.ACC_By_Attn(:,:,SS) = BEHAVE.ACC_By_Attn;
        BEHAVEALL.ACC_By_Col(:,:,SS) = BEHAVE.ACC_By_Col;
        BEHAVEALL.RT_By_AttnandCol(:,:,SS) = BEHAVE.RT_By_AttnandCol;
        
        %% Allocate accuracy
        for chunksizeiterator = 1:n.chunksizes
            % get chunk size
            chunksize = samples.chunksizes(chunksizeiterator); % Choose chunksize
            samplepoints =  chunksize  : samples.slidingwindow : samples.trial; % The points we would hypothetically be calculating this at in realtime
            n.chunks = length(samplepoints);
            
            % Allocate
            ACC_ARRANGED.CCA(n.chunks_max-n.chunks+1:end,:,SS,chunksizeiterator) = ACCURACY_arranged.CCA{chunksizeiterator}(:,1:n.trials);
            ACC_ARRANGED.zscore(n.chunks_max-n.chunks+1:end,:,SS,chunksizeiterator) = ACCURACY_arranged.zscore{chunksizeiterator}(:,1:n.trials);
            
            for HZSTATE = 1:n.train_freq_cond
                ACC_ARRANGED.LDA(n.chunks_max-n.chunks+1:end,:,SS,chunksizeiterator,HZSTATE) = ACCURACY_arranged.LDA.(str.HzState{HZSTATE}){chunksizeiterator}(:,1:n.trials);
                ACC_ARRANGED.KNN(n.chunks_max-n.chunks+1:end,:,SS,chunksizeiterator,HZSTATE) = ACCURACY_arranged.KNN.(str.HzState{HZSTATE}){chunksizeiterator}(:,1:n.trials);
                ACC_ARRANGED.MLP(n.chunks_max-n.chunks+1:end,:,SS,chunksizeiterator,HZSTATE) = ACCURACY_arranged.MLP.(str.HzState{HZSTATE}){chunksizeiterator}(:,1:n.trials);
            end
        end

    end
    
    %% get combined data - Hz state dat
%     if SS ~=30
        for ii = 1:n.HzstateMetrics
            tmp = squeeze(mean(mean(nanmean(ACC_ARRANGED.(str.HzstateMetrics{ii})(:,:,:,end,:),1),2),4)).*100;
            HZSTATEDAT(AA, ii, :, :) = tmp';
            HZSTATEDAT_M(AA, ii, :) = mean(tmp);
            HZSTATEDAT_E(AA, ii, :) = ws_bars(tmp);
            
            tmp = squeeze(mean(nanmean(ACC_ARRANGED.(str.HzstateMetrics{ii})(:,:,:,:,:),1),2)).*100;
            HZSTATEDAT_time(AA, ii, :, :, :) = permute(tmp, [3 2 1]);
            
        end
%     end
    
    %% get combined data - chunksize compare 
    for ii = 1:n.methods
        if ii <3
            DAT = squeeze(mean(nanmean(ACC_ARRANGED.(str.algorithm{ii}),1),2)).*100;
        else
            
            % Get optimal set of frequencies for each person
            tmp = squeeze(mean(mean(nanmean(ACC_ARRANGED.(str.algorithm{ii}),1),2),4));
            [tmp2,Hzstateuse] = max(tmp,[],2);
            
            % get data using optimum
            DAT = NaN(n.subs,n.chunksizes);
            for SS = 1:n.subs
                DAT(SS,:) =  squeeze(mean(nanmean(ACC_ARRANGED.(str.algorithm{ii})(:,:,SS,:,Hzstateuse(SS)),1),2)).*100;
                %                 DAT(SS,:) =  squeeze(mean(nanmean(ACC_ARRANGED.(str.algorithm{ii})(:,:,SS,:,1),1),2)).*100;
            end
        end
        
        ACC_CHUNK(AA, :, ii,:) = DAT';
    end
    
     %% get combined data - chunksize compare variance
    for ii = 1:n.methods
         if ii <3
            DAT = squeeze(var(squeeze(nanmean(ACC_ARRANGED.(str.algorithm{ii}),1)),1));
        else
            tmp = squeeze(mean(mean(nanmean(ACC_ARRANGED.(str.algorithm{ii}),1),2),4));
            [tmp2,Hzstateuse] = max(tmp,[],2);
            
            % get data using optimum
            DAT = NaN(n.subs,n.chunksizes);
            for SS = 1:n.subs
                DAT(SS,:) =  squeeze(var(squeeze(nanmean(ACC_ARRANGED.(str.algorithm{ii})(:,:,SS,:,Hzstateuse(SS)),1)),1));
            end
            
        end
        
        VAR_CHUNK(AA, :, ii,:) = DAT';
    end
end


%% Plot Var summary plot

M = squeeze(mean(mean(VAR_CHUNK,2),4));
E = [ws_bars(squeeze(mean(VAR_CHUNK(1,:,:,:),2))'); ws_bars(squeeze(mean(VAR_CHUNK(2,:,:,:),2))')];


h = figure;
[~,hE] = barwitherr(E',M', 'linewidth', 3);
set(hE, 'CapSize', 5, 'linewidth', 3)
set(gca, 'xticklabel',str.algorithm, 'tickdir', 'out', 'LineWidth', 3)
box('off')


legend(str.traintype, 'location', 'NorthWest')
colormap([1 1 1; 0 0 0])

ylabel('Variance')
xlabel('Algorithm Type')


tit = ['Chunksize Effect variance MIXED'];
title(tit)
saveas(h, [direct.results tit  '.png'])
saveas(h, [direct.results tit  '.eps'], 'epsc')

%% Plot Hzstate results together alltogether.

str.HzstateMetricsMixed = {'LDA - Multi'    'KNN - Multi'    'MLP - Multi'  'LDA - SingleToMulti'    'KNN - SingleToMult'    'MLP - SingleToMult'};
LStyles = {'-' '--'};

h = figure;
hold on;
for AA = 1:2
    for ii = 1:n.HzstateMetrics

        M = squeeze(HZSTATEDAT_M(AA,ii,:));
        E = squeeze(HZSTATEDAT_E(AA,ii,:));
        errorbar(M,E, 'linewidth', 2, 'color', linecolours(ii+2,:), 'CapSize', 12, 'LineStyle', LStyles{AA});
    end
end

xlim([0 5])
% ylim([50 57]);
set(gca, 'xtick', 1:4,  'xticklabel', str.HzState,'tickdir', 'out')

set(gca, 'FontName', 'arial',  'LineWidth', 3)
box('off')

legend(str.HzstateMetrics)

xlabel('Frequency Features Used')
ylabel('Variance across Trials')

tit = ['Frequency contributions variance MIXED'];
title(tit)
saveas(h, [direct.results tit  '.png'])
saveas(h, [direct.results tit  '.eps'], 'epsc')


%% Hzstatedat individual differences

H = figure;
hold on
ALL_Ms = [];
for SS = 1:30
    
    ii = 1;
    AA = 1;
    dat = squeeze(HZSTATEDAT(AA,ii,:, SS));
    [M1, maxi(SS)] = max(dat);
    M2 = min(dat);
    M = [ M1 M2];
    E = zeros(2,1);
    errorbar(M,E, 'linewidth', 2, 'color', linecolours(ii+2,:), 'CapSize', 12, 'LineStyle', LStyles{AA});
    
    ALL_Ms = [ALL_Ms; M];
end


xlim([0 3])
% ylim([50 57]);
% set(gca, 'xtick', 1:4,  'xticklabel', str.HzState,'tickdir', 'out')

set(gca, 'FontName', 'arial',  'LineWidth', 3)
box('off')

% range
% min(ALL_Ms) =    49.1111   47.94
% max(ALL_Ms) =   88.1667   86.94
% mean(ALL_Ms) =   63.4954   60.67
% mean(ALL_Ms) =   9.2785    9.7392
% [~,p,~,stats] = ttest(ALL_Ms(:,1), ALL_Ms(:,2))
% p =   1.7483e-12
%     tstat: 11.6783
%        df: 29
%        sd: 1.3206
[sum(maxi==1) sum(maxi==2) sum(maxi==3) sum(maxi==4)]
% 12    11     3     4
%% stats on Hzstate results
% 3 way anova
% data
ACC = [];

% factors
TRAINTYPE = cell(n.subs*2*3*4,1);
ALGORITHM = cell(n.subs*2*3*4,1);
HZSTATE = cell(n.subs*2*3*4,1);

% create data and variable vectors
n.multistates = 2;
idx = 1:30;
for AA = 1:n.multistates
    for ii = 1:n.HzstateMetrics
        for HH = 1:n.Hzstates
            dat = squeeze(HZSTATEDAT(AA, ii, HH, :));
            
            % data
            ACC = [ACC; dat];
            
            % factors
            TRAINTYPE(idx)  = {str.traintype{AA}};
            ALGORITHM(idx) = {str.HzstateMetrics{ii}};
            HZSTATE(idx) ={str.HzState{HH}};

            idx = idx+30;
        end
    end
end

varnames = {'TRAINTYPE';'ALGORITHM';'HZSTATE'};
anovan(ACC,{TRAINTYPE ALGORITHM HZSTATE},3,3,varnames)

% report - no main effect or interaction with Hzstate - therefore no more
% follow up


%% stats on Hzstate results with time
% 3 way anova
% data
str.windowsize = {'0.25' '0.50' '1.00' '2.00' '4.00'};
ACC = [];

% factors
TRAINTYPE = cell(n.subs*2*3*4*5,1);
ALGORITHM = cell(n.subs*2*3*4*5,1);
HZSTATE = cell(n.subs*2*3*4*5,1);
WINDOWSIZE = cell(n.subs*2*3*4*5,1);

% create data and variable vectors
n.multistates = 2;
idx = 1:30;
for AA = 1:n.multistates
    for ii = 1:n.HzstateMetrics
        for HH = 1:n.Hzstates
            for WW = 1:n.chunksizes
                dat = squeeze(HZSTATEDAT_time(AA, ii, HH, WW, :));
                
                % data
                ACC = [ACC; dat];
                
                % factors
                TRAINTYPE(idx)  = {str.traintype{AA}};
                ALGORITHM(idx) = {str.HzstateMetrics{ii}};
                HZSTATE(idx) = {str.HzState{HH}};
                WINDOWSIZE(idx) = {str.windowsize{WW}};
                
                idx = idx+30;
            end
        end
    end
end

varnames = {'TRAINTYPE';'ALGORITHM';'HZSTATE'; 'WINDOWSIZE'};
anovan(ACC,{TRAINTYPE ALGORITHM HZSTATE WINDOWSIZE},4,3,varnames)

% report - no main effect or interaction with Hzstate - therefore no more
% follow up


%% Get chunk information to determine when this becomes significant

dat = squeeze(mean(mean(ACC_CHUNK,1),3));
M = mean(dat,2);

% - calculate 95% confidince intervals
% E = ws_bars(dat');
E = NaN(size(M));
P = NaN(size(M));
for tt = 1:n.chunksizes
    [~,P(tt),CI,~] = ttest(dat(tt,:), 50);
    E(tt) = diff(CI)./2;
end
    

h = figure;
[~,hE] = barwitherr(E,M, 'linewidth', 3);
set(hE, 'CapSize', 20, 'linewidth', 3)
set(gca, 'xticklabel',time.chunksizes, 'tickdir', 'out', 'LineWidth', 3)
box('off')

ylim([48 60])
colormap([0.9 0.9 0.9])
line([0 6], [50 50], 'color', 'r', 'linewidth', 3) 

ylabel('Chunk Size')
xlabel('Accuracy')

% outcome - use 0.75 seconds onwards. 


tit = ['Chunksize Effect MIXED'];
title(tit)
saveas(h, [direct.results tit  '.png'])
saveas(h, [direct.results tit  '.eps'], 'epsc')

%% Get chunk information to determine when this becomes significant

chunkstouse = 3:5;
dat = squeeze(mean(ACC_CHUNK(:,chunkstouse,:,:),2)); % average across epoch lengths larger than 50
dat = dat(:,:,:);
M = mean(dat,3);
E = NaN(size(M));
for ii = 1:n.methods
    E(:,ii) = ws_bars(squeeze(dat(:,ii,:))');
end

h = figure;
[~,hE] = barwitherr(E',M', 'LineWidth', 3);
set(hE, 'CapSize', 10, 'linewidth', 3)

set(gca, 'xticklabel',str.algorithm, 'tickdir', 'out', 'LineWidth', 3)
box('off')

ylim([48 60])
legend(str.traintype, 'location', 'NorthWest')
colormap([1 1 1; 0 0 0])
line([0 6], [50 50], 'color', 'r', 'linewidth', 3) 

ylabel('Chunk Size')
xlabel('Accuracy')


tit = ['Algorithm effect MIXED'];
title(tit)
saveas(h, [direct.results tit  '.png'])
saveas(h, [direct.results tit  '.eps'], 'epsc')

%% Compare short and long window sizes across traintype

Short =  squeeze(mean(mean(ACC_CHUNK(:,1:3,:,:),2), 3));

M = mean(Short,2);
STD = std(Short');
[~,p,~,stats] = ttest(Short(1,:), Short(2,:));
% p =   7.3828e-08
% tstat: -7.1404
% df: 29
% sd: 1.5129
% M =   [49.6553   51.6275] % multi, singtomult
% STD =    [2.4177    1.4860] % multi, singtomult

Long =  squeeze(mean(mean(ACC_CHUNK(:,4:5,:,:),2), 3));

M = mean(Long,2);
STD = std(Long');
[~,p,~,stats] = ttest(Long(1,:), Long(2,:));
% p =   7.0446e-04
% tstat: 3.7905
% df: 29
% sd: 3.1074
% M =   [56.8928   54.7423] % multi, singtomult
% STD =    [6.0149    4.0991] % multi, singtomult

return
%% Get average trajectory over chunk size increase

dat = squeeze(mean(mean(ACC_CHUNK, 1), 3));
figure; plot(time.chunksizes, mean(dat'))

%  fit with inverse exponential function
x = time.chunksizes;
y =  mean(dat')'; % y(1) = [];

myfittype=fittype('a*(1-exp(-b*(x-c)))',...
'dependent', {'y'}, 'independent',{'x'},'coefficients', {'a' 'b' 'c'});

myfit=fit(x',y,myfittype,'StartPoint',[0.5 0.5 0.5]);


% get maximum
x2 = 0.25:0.25:20;
y2 = myfit.a*(1-exp(-myfit.b*(x2-myfit.c)));

[~,iH] = min(abs(y2 - max(y2)*0.99))
x2(iH)
myfit.c + (log(1-0.99)/-myfit.b)

% Plot results
h= figure; hold on;
x2 = 1:0.25:4;
plot(x2, myfit.a*(1-exp(-myfit.b*(x2-myfit.c))), 'g--')
plot(x,y, 'g-x')


%% stats on windowsize results
% 3 way anova
% data

ACC = [];

% factors
TRAINTYPE = cell(n.subs*2*5*5,1);
ALGORITHM = cell(n.subs*2*5*5,1);
WINDOWSIZE = cell(n.subs*2*5*5,1);

% create data and variable vectors
n.multistates = 2;
idx = 1:30;
for TT = 1:n.multistates
    for AA = 1:n.methods
        for WW = 1:n.chunksizes
            dat = squeeze(ACC_CHUNK(TT,WW,AA,:)); % average across epoch lengths larger than 50
            
            % data
            ACC = [ACC; dat];
            
            % factors
            TRAINTYPE(idx)  = {str.traintype{TT}};
            ALGORITHM(idx) = {str.algorithm{AA}};
            WINDOWSIZE(idx) = {str.windowsize{WW}};
            
            idx = idx+30;
        end
    end
end

varnames = {'TRAINTYPE';'ALGORITHM';'WINDOWSIZE'};
[p,tbl,stats,terms] = anovan(ACC,{TRAINTYPE ALGORITHM WINDOWSIZE},3,3,varnames)

% report - no main effect or interaction with Hzstate - therefore no more
% follow up

% Partial Eta Squared!
% https://www.frontiersin.org/articles/10.3389/fpsyg.2013.00863/full

tbl{5,1} = 'TRAINTYPE_ALGORITHM'
tbl{6,1} = 'TRAINTYPE_WINDOWSIZE'
tbl{7,1} = 'ALGORITHM_WINDOWSIZE'
tbl{8,1} = 'TRAINTYPE_ALGORITHM_WINDOWSIZE'

% Get all the sums of squares
for N = 2:10 
    SSS.(tbl{N,1}) = tbl{N,2}
end

% Get all the sums of squares
for N = 2:8
    partialetasquared.(tbl{N,1}) = SSS.(tbl{N,1}) / (SSS.(tbl{N,1}) + SSS.Error)
end

% partialetasquared =
% 
%   struct with fields:
% 
%                          TRAINTYPE: 0.0013
%                          ALGORITHM: 0.0332
%                         WINDOWSIZE: 0.2799
%                TRAINTYPE_ALGORITHM: 0.0771
%               TRAINTYPE_WINDOWSIZE: 0.0557
%               ALGORITHM_WINDOWSIZE: 0.0967
%     TRAINTYPE_ALGORITHM_WINDOWSIZE: 0.0363

