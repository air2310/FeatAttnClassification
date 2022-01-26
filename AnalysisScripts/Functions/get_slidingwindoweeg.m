function [chunkeeg, chunklabels] = get_slidingwindoweeg(trialeeg,sets)
% get_eeg: Get eeg data for decoding
    %   Inputs:
    %        trialeeg - structure containing trialwise EEG and corresponding labels
    %        set -Structure describing how the decoding analysis will run.
    %   Outputs:
    %       chunkeeg - EEG data for each different sliding window size
    %       chunklabels - labels corresponding to data for each different
    %       sliding window size. 
    
    %% Loop through sliding window sizes
    
    % Preallocate
    chunkeeg = cell(sets.n.chunksizes,1);
    chunklabels = cell(sets.n.chunksizes,1);
    
    % Loop through sliding windows
    for ii_chunk = 1:sets.n.chunksizes
        % Sliding window specific timing
        chunksize = sets.timing.samples.chunksizes(ii_chunk); % Choose chunksize
       
        % Get Chunks
        [DATA, chunklabels{ii_chunk}] = run_chunking(chunksize);
        
        % Zero padding - pad data of each chunksize out to achieve equal spectral resolution.
        padlength = (sets.epoch_chunk{ii_chunk}.n.x - chunksize)/2;
        padding = zeros(padlength, sets.eeg.n_chans, chunklabels{ii_chunk}.n_epochs);
        paddeddata = cat(1, padding, DATA, padding);
        
        % Assign padded data. 
        chunkeeg{ii_chunk} = paddeddata;

    end
    
    %% Nested function run_chunking
    function [DATA, LABELS] = run_chunking(chunksize)
       % Specific timing things for this chunksize
        samplepoints =  chunksize  : sets.timing.samples.slidingwindowstep : sets.timing.samples.trial; % The points we would hypothetically be calculating this at in realtime
        n.chunks = length(samplepoints);
        
        % Preallocate
        n.epochs = n.chunks * sets.n.trials_all;
        DATA = NaN(chunksize, sets.eeg.n_chans, n.epochs);
        LABELS.trial = NaN(n.epochs,1);
        LABELS.chunk = NaN(n.epochs,1);
        LABELS.stimstate = NaN(n.epochs,1);
        LABELS.hzcued = NaN(n.epochs,1);
        LABELS.colcued = NaN(n.epochs,1);
        LABELS.reject_noise = zeros(n.epochs,1);
        LABELS.reject_motionepoch = zeros(n.epochs,1);
        
        counter = 0;
        for TRIAL = 1:sets.n.trials_all
            % Get time chunks to exclude due to motion events
            exclude_samples = [];
            for mm = 1:sets.n.motionevents
                exclude_samples = [exclude_samples trialeeg.motiononsets_relative(TRIAL,mm) :  trialeeg.motiononsets_relative(TRIAL,mm)+sets.timing.samples.motionepochs];
            end
            
            % Cycle through chunks. 
            for CHUNK = 1:n.chunks
                % Get indices
                counter = counter + 1;
                start = samplepoints(CHUNK)- chunksize +1;
                stop = samplepoints(CHUNK);
                
                % Allocate data
                tmp = trialeeg.dat(start:stop, :, TRIAL);
                tmp = detrend(tmp, 'linear');
                tmp = tmp - tmp(1,:);
                DATA(:,:,counter) = tmp; % Each chunk will be used for training.
                
                % Allocate labels
                LABELS.trial(counter) = TRIAL;
                LABELS.chunk(counter) = CHUNK;
                LABELS.stimstate(counter) = trialeeg.labels_stimstate(TRIAL);
                LABELS.hzcued(counter) =  trialeeg.labels_hzcued(TRIAL);
                LABELS.colcued(counter) =  trialeeg.labels_colcued(TRIAL);
                
                % Exclude noisy chunks
                if any(any(abs(tmp)>150)) % 70 exclude noisy chunks from training.
                    LABELS.reject_noise(counter) = 1;
                end
                
                % exclude periods of motion epochs
                if sum(ismember(start:stop, exclude_samples))/chunksize > sets.fraction_motexclude
                   LABELS.reject_motionepoch(counter) = 1;
                end
            end
        end
        
        LABELS.n_epochs = n.epochs; % the total number of epochs for training
    end
end