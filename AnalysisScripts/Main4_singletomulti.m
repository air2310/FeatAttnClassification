%% Main Script for comparison of multiple methods of Real-time, singletrial, attentional selectivity calculation.
% Created by Angela I. Renton on 07/12/2018. Email: angie.renton23@gmail.com
%

%% Clean Up

clear
clc
close all

cd 'E:\angie\RTAttnSelectMethods\Analysis'
%% Settings

for SUB = 1:32
     if ismember(SUB, [7 19])
        continue
     end
    
    close all
    str.SNRState = {'amp' 'SNR'};
    str.HzState = {'Basic' 'Harmonic' 'Alpha' 'AlphaPlusHarmonic'};

    options.collate=0;
    setupSettingsBIDS
    %% Get EEG data and Metadata
    % Here we extract the trials of EEG data:

    getEEG_singlemultiBIDS
    %% Generate CCA Templates
    
    for ATTNSTATE = 1:2
        generateCCATemplates
        
        CCA.CCA_SIN.(str.attnstate{ATTNSTATE}) = CCA_SIN;
        CCA.CCA_ERP.(str.attnstate{ATTNSTATE}) = CCA_ERP;
        CCA.trial_harm.(str.attnstate{ATTNSTATE}) = trial_harm;
    end
   
    %% Cycle through chunks of various data chunk lengths
    
    for chunkSizeIterator = 1:n.chunksizes
        tic
        chunksize = samples.chunksizes(chunkSizeIterator); % Choose chunksize
        
        samplepoints =  chunksize  : samples.slidingwindow : samples.trial; % The points we would hypothetically be calculating this at in realtime
        n.chunks = length(samplepoints);
        
        % Display current loop status
        disp(['Epoch Length: ' num2str(time.chunksizes(chunkSizeIterator)) ' s, ' num2str(n.chunks) ' chunks'])
        
        %% Sliding window data extraction
        % We don't want to save the data separately for every data size, so extract
        % this separately for every chunk length
        %
        % Extract the various chunks of the size we're working on at the moment:
        clear DATA LABELS DATA2
        
        for ATTNSTATE = 1:2
            % preallocate data and labels for this chunksize
            DATA.(str.attnstate{ATTNSTATE}) = NaN(chunksize, n.channels, n.chunks * n.trials);
            LABELS.(str.attnstate{ATTNSTATE}) = NaN( n.chunks * n.trials,4);
            
            counter = 0;
            for TRIAL = 1:n.trials
                for CHUNK = 1:n.chunks
                    counter = counter + 1;
                    start = samplepoints(CHUNK)-chunksize +1;
                    stop = samplepoints(CHUNK);
                    
                    DATA.(str.attnstate{ATTNSTATE})(:,:,counter) = TRIAL_EEG.(str.attnstate{ATTNSTATE})(start:stop, :, TRIAL); % Each chunk will be used for training.
                    
                    LABELS.(str.attnstate{ATTNSTATE})(counter,1) = LABELS_EEG.(str.attnstate{ATTNSTATE})(TRIAL);
                    LABELS.(str.attnstate{ATTNSTATE})(counter,2) = TRIAL;
                    LABELS.(str.attnstate{ATTNSTATE})(counter,3) = CHUNK;
                    
                    if any(any(abs(TRIAL_EEG.(str.attnstate{ATTNSTATE})(start:stop, :, TRIAL))>150)) % exclude noisy chunks from training.
                        LABELS.(str.attnstate{ATTNSTATE})(counter,4) = 0;
                    else
                        LABELS.(str.attnstate{ATTNSTATE})(counter,4) = 1;
                    end
                    
                end
            end
            
            n.epochs = counter; % the total number of epochs for training
            
            %% Get SSVEPs
            % Perform FFTs on each chunk to get the amplitude at frequencies of interest:
            
            % Zero padding - pad chunk of data out to achieve adequete spectral resolution.
            padlength = (n.x-chunksize)/2;
            padding = zeros(padlength, n.channels, n.epochs);
            
            DATA2.(str.attnstate{ATTNSTATE}) = cat(1, padding, DATA.(str.attnstate{ATTNSTATE}), padding);
            
            % run FFT
            tmp = abs( fft( DATA2.(str.attnstate{ATTNSTATE})(:,:,~~LABELS.(str.attnstate{ATTNSTATE})(:,4)) ) )/n.x;
            tmp(2:end-1,:,:) = tmp(2:end-1,:,:)*2;
            
            % Plot frequency spectrum
            figure;
            plot(f, mean(mean(tmp,2),3))
            xlim([5 16])
            line([Hz(1) Hz(1); Hz(2) Hz(2)]', [get(gca, 'ylim');get(gca, 'ylim')]', 'color', 'r')
            ylabel('FFT amp. (µV)'); xlabel('Frequency (Hz)')
            title(['Mean Freq. spectrum after zeropadding , Epoch Length: ' num2str(time.chunksizes(chunkSizeIterator)) ', ' str.attnstate{ATTNSTATE} ])
            
        end
        %% Cycle through Frequency options
        
        % HzState = {'Basic' 'Harmonic' 'Alpha' 'AlphaPlusHarmonic'}
        %  Hz = [6 7.5 12 15];
        for HZSTATE = 1:4
            
            for ATTNSTATE = 1:2
                % run FFT
                tmp = abs( fft( DATA2.(str.attnstate{ATTNSTATE})  ) )/n.x;
                tmp(2:end-1,:,:) = tmp(2:end-1,:,:)*2;
                
                % Take SSVEPs at the correct frequency
                switch HZSTATE
                    case 1
                        AMP_RAW.(str.attnstate{ATTNSTATE})  = tmp(idx.Hz(1:2)  ,:,:);
                    case 2
                        AMP_RAW.(str.attnstate{ATTNSTATE})  = tmp(idx.Hz  ,:,:);
                    case 3
                        alpha = mean( tmp(idx.Hz_alpha  ,:,:),1);
                        AMP_RAW.(str.attnstate{ATTNSTATE})  = cat(1, tmp(idx.Hz(1:2)  ,:,:), alpha);
                    case 4
                        alpha = mean( tmp(idx.Hz_alpha  ,:,:),1);
                        AMP_RAW.(str.attnstate{ATTNSTATE})  = cat(1, tmp(idx.Hz  ,:,:), alpha);
                end
            end
            
            
            %% Start Machine Learning things!
            %% 1. z-scoreing
            %
            
            if HZSTATE==1
                % Z-score SSVEPs
                
                AMPALL = NaN(2,n.epochs);
                
                tmp = squeeze(nanmean(AMP_RAW.MultiFreq,2));
                
                % Take mean and std on chunks that weren't corrupted by noise
                MEAN = mean(tmp(:,~~LABELS.MultiFreq(:,4)),2);
                STD = std(tmp(:,~~LABELS.MultiFreq(:,4))')';
                
                % Z-score
                AMPALL(1,:) = (tmp(1,:)-MEAN(1))./STD(1);
                AMPALL(2,:) = (tmp(2,:)-MEAN(2))./STD(2);
                
                %                 AMPALL(1,:) = zscore(tmp(1,:));
%                 AMPALL(2,:) = zscore(tmp(2,:));
                
                % switch ordering from [6Hz, 7.5Hz] to [attended, unattended].
                indices = [1 2; 2 1];
                for ii = 1:n.epochs
                    AMPALL(:,ii) = AMPALL(indices( LABELS.MultiFreq(ii,1),:), ii) ;
                end
                
                % Plot bar graph of SSVEPs
                M = mean(mean(AMPALL,2),3);
                figure;
                bar(M)
                xlim([0 3])
                set(gca, 'xTickLabel', {'Attended' 'Unattended'})
                xlabel('Feature')
                ylabel('standardised SSVEP amp.')
                title(['Mean SSVEP amplitudes after zeropadding , Epoch Length: ' num2str(time.chunksizes(chunkSizeIterator)) ])
                
                % Get Features
                FEATURE = AMPALL;%squeeze(mean(AMPALL,2)); % average across channels
                FEATURE = FEATURE(1,:) - FEATURE(2,:);
                
                % Get accuracy
                ACCURACY_ALL.zscore{chunkSizeIterator} = sign(FEATURE)==1;
                TMPACC = sum( ACCURACY_ALL.zscore{chunkSizeIterator})/length( ACCURACY_ALL.zscore{chunkSizeIterator});
            end
            
            %% 2-4. Linear Discriminant Analysis, K-Nearest Neighbour, and Support Vector Machine
            clear FEATURE
            for ATTNSTATE = 1:2
                % get feature vector
                tmp = AMP_RAW.(str.attnstate{ATTNSTATE})(1,:,:);
                for HH = 2:size(AMP_RAW.(str.attnstate{ATTNSTATE}),1)
                    tmp = cat(2, tmp, AMP_RAW.(str.attnstate{ATTNSTATE})(HH,:,:));
                end
                FEATURE.(str.attnstate{ATTNSTATE}) = squeeze(tmp);
            end
            
            % Preallocate
            ACCURACY_ALL.LDA.(str.HzState{HZSTATE}){chunkSizeIterator} = NaN(n.epochs,1);
            ACCURACY_ALL.KNN.(str.HzState{HZSTATE}){chunkSizeIterator} = NaN(n.epochs,1);
            
            % Organise data
            inputs_train = FEATURE.SingleFreq';
            labels_train = LABELS.SingleFreq(:,1) ;
            
            inputs_train(~LABELS.SingleFreq(:,4),:)=[];
            labels_train(~LABELS.SingleFreq(:,4)) = []; % eliminate noisy trials from training
            
            inputs_test = FEATURE.MultiFreq';
            labels_test = LABELS.MultiFreq(:,1);
            
            % Train Classifiers
            classifier_LDA = fitcdiscr( inputs_train, labels_train);
            classifier_KNN = fitcknn( inputs_train, labels_train);
%             classifier_SVM = fitcsvm( inputs_train, labels_train);
            
            % Test
            y_predict = predict( classifier_LDA, inputs_test );
            ACCURACY_ALL.LDA.(str.HzState{HZSTATE}){chunkSizeIterator} = y_predict == labels_test;
            
            y_predict = predict( classifier_KNN, inputs_test );
            ACCURACY_ALL.KNN.(str.HzState{HZSTATE}){chunkSizeIterator} = y_predict == labels_test;
            
%             y_predict = predict( classifier_SVM, inputs_test );
%             tmp = y_predict == labels_test;
            %% 5. Multi-layer Perceptron

            % Preallocate
            ACCURACY_ALL.MLP.(str.HzState{HZSTATE}){chunkSizeIterator} = NaN(n.epochs,1);
            
            % Organise data
            
            inputs_train = FEATURE.SingleFreq';
            labels_train = LABELS.SingleFreq(:,1);
            
            inputs_train(~LABELS.SingleFreq(:,4),:)=[];
            labels_train(~LABELS.SingleFreq(:,4)) = []; % eliminate noisy trials from training
             
            inputs_test = FEATURE.MultiFreq';
            labels_test = LABELS.MultiFreq(:,1);
                
            % Labels need to be stored slightly differently for MLP
            tt = zeros(2, length(labels_train));
            tt(1,labels_train==1) = 1; tt(2,labels_train==2) = 1;
            
            % create empty net
            net = patternnet(hiddenLayerSize, trainFcn, performFcn);
            
            
            %## Train the Network
            [net,tr] = train(net,inputs_train',tt);
            
            % Test
            y_predict = net(inputs_test');
            ACCURACY_ALL.MLP.(str.HzState{HZSTATE}){chunkSizeIterator} = vec2ind(y_predict)'  == labels_test;
                

            sum(ACCURACY_ALL.MLP.(str.HzState{HZSTATE}){chunkSizeIterator})/length(ACCURACY_ALL.MLP.(str.HzState{HZSTATE}){chunkSizeIterator})
            
            %% Finish cycling through frequency options
            
        end
        %% Filter-Bank Canonical Correlation Analysis
        
        % preallocate
        ACCURACY_arranged.CCA{chunkSizeIterator} = NaN(n.chunks, n.trials);
        
        for FOLD = 1:n.trials % leave one out
            RHO = NaN(n.chunks, n.Hz_main);
            for CHUNK = 1:n.chunks % for each chunk of data
                % Select a piece of trial
                start = samplepoints(CHUNK)-chunksize +1;
                stop = samplepoints(CHUNK);
                epoch = start:stop;
                
                for FF2 = 1:n.Hz_main
                    
                    rho_use = zeros(1,5);
                    
                    for HH = 1:n.harmonics
                        % Get the data and templates we'll use
                        erp_cca = CCA.CCA_ERP.SingleFreq(epoch,HH,FF2,FOLD);
                        synth_cca = CCA.CCA_SIN.SingleFreq(epoch,HH,FF2,FOLD);
                        
                        trial2classify_long = CCA.trial_harm.MultiFreq( epoch, :,FOLD, HH );
                        trial2classify = mean( trial2classify_long, 2);
                        
                        if ~all( trial2classify(:) == 0 )
                            
                            % ------- canncor
                            
                            % -- test(X) V synth(Y)
                            [WXyX,WXyY,~,~,~] = canoncorr(trial2classify, synth_cca);
                            [~,~,r,~,~] = canoncorr(trial2classify_long, synth_cca);
                            rho_tmp(1) = max(r);
                            
                            % -- test(X) V train(x)
                            [WXxX,WXxx,~,~,~] = canoncorr(trial2classify, erp_cca);
                            [~,~,r,~,~] = canoncorr(trial2classify_long, erp_cca);
                            rho_tmp(2) = max(r);
                            
                            % -- train(x) V synth(Y)
                            [WxYx,WxYY,~,~,~] = canoncorr(trial2classify, erp_cca);
                            
                            % -- test(X) V train(x) -- Weight(test(X) V synth(Y))
                            X = trial2classify; Y = erp_cca;
                            U = (X - repmat(mean(X),size(X,1),1))*WXyX;
                            V = (Y - repmat(mean(Y),size(X,1),1))*WXyY;
                            r = corr(U,V);
                            rho_tmp(3) = max(diag(r));
                            
                            % -- test(X) V train(x) -- Weight(train(x) V synth(Y))
                            X = trial2classify; Y = erp_cca;
                            U = (X - repmat(mean(X),size(X,1),1))*WxYx;
                            V = (Y - repmat(mean(Y),size(X,1),1))*WxYY;
                            r = corr(U,V);
                            rho_tmp(4) = max(diag(r));
                            
                            % -- train(X) V train(x) -- Weight(train(x) V test(x))
                            X = erp_cca; Y = erp_cca;
                            U = (X - repmat(mean(X),size(X,1),1))*WXxX;
                            V = (Y - repmat(mean(Y),size(X,1),1))*WXxx;
                            r = corr(U,V);
                            rho_tmp(5) = max(r);
                            
                        else
                            rho_tmp(1:5) = 0;
                        end
                        
                        %% create weighted combinations
                        
                        div = HH;
                        rho_use(1) = rho_use(1) + (1/div)*(rho_tmp(1)^2);
                        rho_use(2) = rho_use(2) + (1/div)*(rho_tmp(2)^2);
                        rho_use(3) = rho_use(3) + (1/div)*(rho_tmp(3)^2);
                        rho_use(4) = rho_use(4) + (1/div)*(rho_tmp(4)^2);
                        rho_use(5) = rho_use(5) + (1/div)*(rho_tmp(5)^2);
                        
                        
                    end
                    
                    % -- combine different R values
                    RHO(CHUNK, FF2) = ...
                        (sign(rho_use(1))*rho_use(1)^2)+...
                        (sign(rho_use(2))*rho_use(2)^2)+...
                        (sign(rho_use(3))*rho_use(3)^2)+...
                        (sign(rho_use(4))*rho_use(4)^2)+...
                        (sign(rho_use(5))*rho_use(5)^2);
                    
                end
            end
            
            [~,idx.tmp]= max(RHO, [],2);
            ACCURACY_arranged.CCA{chunkSizeIterator}(:,FOLD) = idx.tmp == LABELS_EEG.MultiFreq(FOLD);
        end
        
        ACCURACY_ALL.CCA{chunkSizeIterator} = ACCURACY_arranged.CCA{chunkSizeIterator}(:);
        
        %% Rearrance all the accuracy outputs by trial and chunk
        
        for MM = 1:n.methods-1
            if any(strcmp(methods{MM}, {'LDA'    'KNN'    'MLP'}))
                
                for HZSTATE = 1:4
                    ACCURACY_arranged.(methods{MM}).(str.HzState{HZSTATE}){chunkSizeIterator} = NaN(n.chunks, n.trials);
                end
            else
                ACCURACY_arranged.(methods{MM}){chunkSizeIterator} = NaN(n.chunks, n.trials);
            end
        end
        
        for TRIAL = 1:n.trials
            for CHUNK = 1:n.chunks
                idx.tmp = LABELS.MultiFreq(:,2) == TRIAL & LABELS.MultiFreq(:,3) == CHUNK;
                for MM = 1:n.methods-1
                    if any(strcmp(methods{MM}, {'LDA'    'KNN'    'MLP'}))
                        
                        for HZSTATE = 1:4
                            ACCURACY_arranged.(methods{MM}).(str.HzState{HZSTATE}){chunkSizeIterator}(CHUNK, TRIAL) = ACCURACY_ALL.(methods{MM}).(str.HzState{HZSTATE}){chunkSizeIterator}(idx.tmp);
                        end
                    else
                        ACCURACY_arranged.(methods{MM}){chunkSizeIterator}(CHUNK, TRIAL) = ACCURACY_ALL.(methods{MM}){chunkSizeIterator}(idx.tmp);
                    end
                end
            end
        end
        
        %% Plot the results by decoding method
        
        h = figure;
        for MM = 1:n.methods
            %% Plot result
            subplot(ceil(n.methods/2), 2,MM)
            if any(strcmp(methods{MM}, {'LDA'    'KNN'    'MLP'}))
                
                imagesc(samplepoints./fs, 1:n.trials,   ACCURACY_arranged.(methods{MM}).(str.HzState{1}){chunkSizeIterator}')
                
                title([methods{MM} ...
                    num2str(round(mean(ACCURACY_ALL.(methods{MM}).(str.HzState{1}){chunkSizeIterator})*100)) '% ' ...
                    num2str(round(mean(ACCURACY_ALL.(methods{MM}).(str.HzState{2}){chunkSizeIterator})*100)) '% '...
                    num2str(round(mean(ACCURACY_ALL.(methods{MM}).(str.HzState{3}){chunkSizeIterator})*100)) '% '...
                    num2str(round(mean(ACCURACY_ALL.(methods{MM}).(str.HzState{4}){chunkSizeIterator})*100)) '% '...
                    ])
            else
                imagesc(samplepoints./fs, 1:n.trials,   ACCURACY_arranged.(methods{MM}){chunkSizeIterator}')
                title([methods{MM} ' decoding, ACC = ' num2str(mean(ACCURACY_ALL.(methods{MM}){chunkSizeIterator})*100) '%'])
            end
            ylabel('Trial')
            xlabel('Time (s)')
            colormap(bone)
        end
        tit = [ str.sub ' SingletoMulti Epoch Length ' num2str(time.chunksizes(chunkSizeIterator))];
        suptitle(tit)
        saveas(h, [direct.results tit '.png'])
        %% Finish "Cycle through chunks of various data chunk lengths"  loop
        
        %     return
        toc
    end
    %% Collate Results
    
    samplepoints =  samples.chunksizes(1)  : samples.slidingwindow : samples.trial; % The points we would hypothetically be calculating this at in realtime
    CLIM = [50 100];
    
    
    for MM = 1:n.methods
        
        if any(strcmp(methods{MM}, {'LDA'    'KNN'    'MLP'}))
            for HZSTATE = 1:4
                % Organise mean results
                dat.(methods{MM}).(str.HzState{HZSTATE}) = NaN(size(ACCURACY_arranged.(methods{MM}).(str.HzState{HZSTATE}){1},1), n.chunksizes);
                
                for ii = 1:n.chunksizes
                    tmp = size(ACCURACY_arranged.(methods{MM}).(str.HzState{HZSTATE}){1},1)-size(ACCURACY_arranged.(methods{MM}).(str.HzState{HZSTATE}){ii},1);
                    dat.(methods{MM}).(str.HzState{HZSTATE})(tmp+1:end,ii) = mean(ACCURACY_arranged.(methods{MM}).(str.HzState{HZSTATE}){ii},2).*100;
                    dat.(methods{MM}).(str.HzState{HZSTATE})(tmp+1:end,ii) = smooth(dat.(methods{MM}).(str.HzState{HZSTATE})(tmp+1:end,ii) ,3);
                end
                
                % plot Mean
                h= figure;
                imagesc(samplepoints./fs, 1:6,   dat.(methods{MM}).(str.HzState{HZSTATE})', CLIM)
                set(gca, 'yticklabels', time.chunksizes, 'tickdir', 'out')
                xlabel('Time in trial (s)')
                ylabel('Chunk size (s)')
                c=colorbar;
                title(c, 'ACC (%)');
                title([ str.sub ' ' str.attnstate{ATTNSTATE} ' ' methods{MM} ' ' str.HzState{HZSTATE}]);
                colormap(viridis)
                saveas(h, [direct.results str.sub ' SingletoMulti ' methods{MM} ' ' str.HzState{HZSTATE} ' average accuracy.png'])
                
            end
        else
            % Organise mean results
            dat.(methods{MM}) = NaN(size(ACCURACY_arranged.(methods{MM}){1},1), n.chunksizes);
            
            for ii = 1:n.chunksizes
                tmp = size(ACCURACY_arranged.(methods{MM}){1},1)-size(ACCURACY_arranged.(methods{MM}){ii},1);
                dat.(methods{MM})(tmp+1:end,ii) = mean(ACCURACY_arranged.(methods{MM}){ii},2).*100;
                dat.(methods{MM})(tmp+1:end,ii) = smooth(dat.(methods{MM})(tmp+1:end,ii) ,3);
            end
            
            % plot Mean
            h= figure;
            imagesc(samplepoints./fs, 1:6,   dat.(methods{MM})', CLIM)
            set(gca, 'yticklabels', time.chunksizes, 'tickdir', 'out')
            xlabel('Time in trial (s)')
            ylabel('Chunk size (s)')
            c=colorbar;
            title(c, 'ACC (%)');
            title([ str.sub ' SingletoMulti ' methods{MM}]);
            colormap(viridis)
            saveas(h, [direct.results str.sub ' ' str.attnstate{ATTNSTATE} ' ' methods{MM} ' average accuracy.png'])
        end
    end
    
    %% Save results
    
    save([direct.results str.sub 'SingletoMulti results.mat'], 'ACCURACY_ALL', 'ACCURACY_arranged', 'dat')
    
end