%% Filter settings
n.harmonics = 2;
%% ----- filter settings

ORDER = 4;

% ----- create broad-band harmonic filters for trials
freq_lowpass = [ 8*1 8*2  ];
freq_highpass = [ 5.5*1 5.5*2  ];

% ----- notch filter
[ Xnotch, Ynotch ] = butter( ORDER, ( [48 51] )./(fs/2), 'stop');

% ----- create narrow-band harmonic filters for erps
Xl_erp = cell(n.harmonics,n.Hz); Yl_erp = cell(n.harmonics,n.Hz);
Xh_erp = cell(n.harmonics,n.Hz); Yh_erp = cell(n.harmonics,n.Hz);
Xlowpass = cell(n.harmonics,1); Ylowpass = cell(n.harmonics,1);
Xharm = cell(n.harmonics,1); Yharm = cell(n.harmonics,1);

for HH = 1:n.harmonics
    
    % ----- create narrow-band harmonic filters for erps
    
    for FF = 1:n.Hz
        [ Xl_erp{HH,FF}, Yl_erp{HH,FF} ] = butter(ORDER, HH*(Hz(FF)+0.2)./(fs/2), 'low'); % Define the filter
        [ Xh_erp{HH,FF}, Yh_erp{HH,FF} ] = butter(ORDER, HH*(Hz(FF)-0.2)./(fs/2), 'high'); % Define the filter
    end
    
    % ----- create broad-band harmonic filters for trials
    
    [ Xlowpass{HH}, Ylowpass{HH} ] = butter(ORDER, (freq_lowpass(HH))./(fs/2), 'low');
    [ Xharm{HH}, Yharm{HH} ] = butter(ORDER, (freq_highpass(HH) )./(fs/2), 'high');
    
end


%% Filter Trial data
if ~exist([direct.results str.sub str.attnstate{ATTNSTATE} 'CCATemplates2.mat'], 'file')
    
    % Some specific numbers
    n.trials_block = min(sum([LABELS_EEG==1 LABELS_EEG==2])) -1;
    epoch = 1:n.x_long;
    
    % preallocate
    CCA_ERP = NaN(n.x_long, n.Hz_main, n.harmonics, n.trials);
    CCA_SIN = NaN(n.x_long, n.Hz_main, n.harmonics, n.trials);
    
    % -- Filter Trials by condition
    trial_harm = NaN( n.x_long, n.channels, n.trials, n.harmonics );
    
    for HH = 1:n.harmonics
        for CC = 1:n.channels
        tmp = filtfilt( Xnotch, Ynotch, squeeze(TRIAL_EEG(:,CC,:)) ); % notch
        tmp = filtfilt( Xlowpass{HH}, Ylowpass{HH}, tmp );
        trial_harm(:,CC,:,HH) = filtfilt( Xharm{HH}, Yharm{HH}, tmp);
        end
    end
    % -- create sinusoidal templates
    
    shiftperiod = ( 0 : 0.1 : 1.9 ) * pi; % change back to 1.9 pi
    n.shifts = length( shiftperiod );
    templates_all = NaN( n.x_long, n.harmonics, n.shifts, n.Hz );
    
    for FF = 1:n.Hz_main
        for HH = 1 : n.harmonics
            for SS = 1:length(shiftperiod)
                templates_all(:,HH,SS,FF) = sin( HH*2*pi*Hz(FF)*t_long  + shiftperiod(SS) ); % contstruct signal
            end
        end
    end
    
    %% ----- Fold
    for FOLD = 1:n.trials
        disp(FOLD)
        labelstmp = LABELS_EEG;
        labelstmp(FOLD) = NaN; % exclude current trial
        
        % -- Get ERPs per condition
        erp2use = NaN(n.x_long, 2);
        erp2use(:,1) = nanmean(nanmean(TRIAL_EEG2(:,:,labelstmp==1),3),2);
        erp2use(:,2) = nanmean(nanmean(TRIAL_EEG2(:,:,labelstmp==2),3),2);
        for FF = 1:2
            for HH = 1 : n.harmonics
                tmp =                   filtfilt( Xl_erp{HH,FF}, Yl_erp{HH,FF}, erp2use(:,FF)); % Apply the Butterworth filter
                erp2use_cca(:,HH,FF) =  filtfilt( Xh_erp{HH,FF}, Yh_erp{HH,FF}, tmp); % Apply the Butterworth filter
            end
        end
        
        
        %% ----- correlate trials and sinusoidal templates ~
        
        RHO = cell( n.Hz_main, 1);
        for FF = 1:n.Hz_main
            TRIAL_HARM =  trial_harm(:,:,labelstmp==FF,:) ;
            RHO{FF,1} = sinusoid_canoncorr2( n, TRIAL_HARM, templates_all, epoch );
        end
        
        % ----- reshape into a 5D array
        
        rho = NaN( n.shifts, n.harmonics, n.Hz_main, n.trials_block, n.Hz_main );
        
        for FF = 1:n.Hz_main
            rho( :, :, :, :, FF) = RHO{FF}(:,:,:,1:n.trials_block);
        end
        
        
        %% ----- determine sinusoid phase (Tau) based on CCA classification
        
        maxR = NaN( n.harmonics, n.Hz_main, n.trials_block );
        correct = NaN( n.harmonics, n.Hz_main, n.trials_block );
        accuracy = NaN( n.harmonics, n.Hz_main, n.trials_block );
        Tau = NaN( n.Hz_main, n.harmonics );
        
        for SS = 1:n.shifts
            for FF = 1:n.Hz_main
                for HH = 1:n.harmonics
                    for TT = 1:n.trials_block
                        R = rho(SS,HH,:,TT,FF);
                        [~,maxR(HH,SS,TT)] = max(R);
                        correct(HH,SS,TT) = FF;
                    end
                    
                    diffR = squeeze(maxR(HH,SS,:))-squeeze(correct(HH,SS,:));
                    diffR = diffR(:);
                    accuracy(HH,SS,FF) = sum(diffR==0)/length(diffR);
                    
                end
            end
        end
        
        for FF = 1:n.Hz_main
            for HH = 1:n.harmonics
                [~,i] = max(accuracy(HH,:,FF));
                Tau(FF,HH) = shiftperiod(i);
            end
        end
        
        
        %% ------ create sinusoids with Tau
        
        template_Synth = NaN(n.x_long,n.harmonics,n.Hz_main);
        
        for FF = 1:n.Hz_main
            for HH = 1:n.harmonics
                template_Synth(:,HH,FF) = sin( HH*2*pi*Hz(FF)*t_long  + Tau(FF,HH) ); % contstruct signal
            end
        end
        
        % plot
        %     h = figure; hold on;
        %     plot( template_Synth(:,2,1))
        %     plot( erp2use_cca(:,2,1))
        
        % Store templates for later
        CCA_ERP(:,:,:,FOLD) = erp2use_cca;
        CCA_SIN(:,:,:,FOLD) = template_Synth;
        
    end
    
    save([direct.results str.sub str.attnstate{ATTNSTATE} 'CCATemplates2.mat'], 'CCA_SIN', 'CCA_ERP', 'trial_harm')
    
else
    load([direct.results str.sub  str.attnstate{ATTNSTATE} 'CCATemplates2.mat'])
end