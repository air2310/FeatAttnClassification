function trialeeg = get_eeg(sets)
% get_eeg: Get eeg data for decoding
    %   Inputs:
    %        set -Structure describing how the decoding analysis will run.
    %   Outputs:
    %       trialeeg - structure containing trialwise EEG and corresponding labels
    
%% Get EEG data
% Check existence of trialeeg .mat file
if exist(sets.direct.filename.trialeeg,'file') == 2
    % Load saved data
    load(sets.direct.filename.trialeeg)
    
else
    % Load an process EEG data from raw
    % Begin by loading EEG data
    eeg = load_eeg(0);
    
    % Then get trial-by-trial EEG Data
    trialeeg = extract_trialeeg(eeg);
    
    % Save trialeeg for next time
    save(sets.direct.filename.trialeeg, 'trialeeg')
end

%% Nested function to load EEG data from raw
    function [eeg] = load_eeg(plotopts)
    % load_eeg: Load eeg data
    %   Inputs:
    %        plotopts - logical describing whether or not to plot data. 
    %   Outputs:
    %       eeg - structure containing:
    %             eeg.dat = raw EEG data
    %             eeg.trigs.value = values of triggers
    %             eeg.trigs.latency = samples in EEG data that correspond with
    %             trigger values.
    % Load dataset using fieldtrip
    cfg            = [];
    cfg.dataset    = sets.direct.filename.eeg;
    cfg.continuous = 'yes';
    cfg.channel    = 'all';
    data           = ft_preprocessing(cfg);

    % Extract EEG and triggers
    % Assign data
    EEG = data.trial{1}(1:sets.eeg.n_chans,:)';
    TRIG = data.trial{1}(end,:)';

    % Get Triggers
    tmp = [0; diff(TRIG)];
    LATENCY = find(tmp~=0);
    TYPE = TRIG(LATENCY);

    % Plot Triggers
    if plotopts
        h = figure('visible','on');
        hold on;
        plot(TRIG)
        stem(LATENCY, TYPE)
        xlabel('Sample number')
        ylabel('Trigger Amp.')
        title('Triggers')
    end
    
    % Correction for Subject 15
    % an electrode became corrupted mid-experiment. This was replaced with the
    % average of the remaining channels. 
    if strcmp(sets.str.sub, 'S15')
        EEG(1811000:end,4) = mean(EEG(1811000:end,[1 2 3 5]),2);
    end

    % export
    eeg.dat = EEG;
    eeg.trigs.value = TYPE;
    eeg.trigs.latency = LATENCY;
    end

%% Nested function to get EEG trials and associated labels

    function [trialeeg] = extract_trialeeg(eeg)
        % Begin by fetching the indices (in sample space) of the trial and
        % motion epoch onsets
        [~, motiononsets_relative, idx_trials] = get_motiononsets(eeg);
        
        % Get the labels that align with these trials
        labels_stimstate = NaN(sets.n.trials_all,1);
        labels_hzcued = NaN(sets.n.trials_all,1);
        labels_colcued = NaN(sets.n.trials_all,1);
        for TRIAL = 1:sets.n.trials_all
            trig = eeg.trigs.value(idx_trials(TRIAL));
            labels_hzcued(TRIAL) = ismember(trig, sets.trig.trial(2,:,:))+ 1; %1 - 6, 2 - 7.5
            labels_colcued(TRIAL) = ismember(trig, sets.trig.trial(:,2,:))+ 1; %1 - black, 2 - white
            labels_stimstate(TRIAL) = ismember(trig, sets.trig.trial(:,:,2))+ 1; %1 - multifreq, 2 - singlefreq
        end
        
        % Check counter-balancing.
        for ss = 1:2 % for each stimstate
            for cc = 1:2 % and each colour
                idx=labels_stimstate==ss & labels_colcued==cc;
                if sum(labels_hzcued(idx)==1) ~= sum(labels_hzcued(idx)==2)
                    error('imbalanced trial numbers across frequency conditions! check on this!')
                end
            end
        end
        
        for ss = 1:2 % for each stimstate
            for hh = 1:2 % and each frequency
                idx=labels_stimstate==ss & labels_hzcued==hh;
                if sum(labels_colcued(idx)==1) ~= sum(labels_colcued(idx)==2)
                    error('imbalanced trial numbers across colour conditions! check on this!')
                end
            end
        end
        
        % Epoch data to trials
        start = eeg.trigs.latency(idx_trials)+sets.epoch_long.lim.x(1);
        stop = eeg.trigs.latency(idx_trials)+sets.epoch_long.lim.x(2);
        
        TRIAL_EEG = NaN(sets.epoch_long.n.x, sets.eeg.n_chans, sets.n.trials_all);
        for TRIAL = 1:sets.n.trials_all
            
            tmp = eeg.dat(start(TRIAL):stop(TRIAL),:);
            tmp = detrend(tmp, 'linear');
            tmp = tmp - tmp(1,:);
            
            TRIAL_EEG(:,:,TRIAL) = tmp;
            
        end
        
        % export data
        trialeeg.dat = TRIAL_EEG;
        trialeeg.labels_stimstate =labels_stimstate;
        trialeeg.labels_hzcued =labels_hzcued;
        trialeeg.labels_colcued =labels_colcued;
        trialeeg.motiononsets_relative =motiononsets_relative;
    end

%% Nested function to get motion onset times
    function [motiononsets, motiononsets_relative, idx_trials] = get_motiononsets(eeg)
        % Motion onset triggers are missing for some participants, so we'll get
        % the motion onsets back from the behavioural data.
        % additional metadata
        % 1. get trial onsets and check to make sure there are 160.
        COND = sets.trig.trial(:); % trig.motion = NaN(n.Hz_main,n.cols,n.attnstates, n.dirs);
        idx_trials = find(ismember(eeg.trigs.value, COND));
        
        % 2. Load behavioural data.
        load(sets.direct.filename.behave, 'DATA', 'mon')
        
        % 3. Checks and balances
        % corrections for specific subjects who had to restart the
        % experiment midway
        if strcmp(sets.str.sub, 'S2') % delete repeated trial due to restart;
            idx_trials(58) = [];
        end
        if strcmp(sets.str.sub, 'S5')% delete repeated trial due to restart; delete last couple trials!
            idx_trials(111) = [];
        end
        
        % Make sure we have the right number of triggers
        if length(idx_trials) ~= sets.n.trials_all
            error('Missing trial triggers! Reconstruction of motion onsets will not work!')
        end
        
        % make sure triggers are correct
        cond = sets.trig.trial(:,:,2);
        tmp = [ismember(eeg.trigs.value(idx_trials), cond(:))+1, DATA.ATTENTIONCOND];
        tmp = tmp(:,1)-tmp(:,2);
        if any(tmp~=0)
            error('incorrect triggers somewhere!')
        end
        
        % 4. Loop through trials to set motion onset times.
        motiononsets = NaN(sets.n.trials_all, sets.n.motionevents); % Since the begining of the EEG recording
        motiononsets_relative = NaN(sets.n.trials_all, sets.n.motionevents);% Since the begining of the trial (since trig.trial)
        for TRIAL = 1:sets.n.trials_all
            tmp = DATA.moveframe__Attd_UnAttd{TRIAL,1};
            tmp = (tmp./mon.ref).*sets.eeg.fs; % switch from monitor refresh time to EEG sample time. 
            motiononsets_relative(TRIAL,:) = tmp;
            motiononsets(TRIAL,:) =  tmp + eeg.trigs.latency(idx_trials(TRIAL)); % add the start time of this trial in the recording.
        end
    end
end