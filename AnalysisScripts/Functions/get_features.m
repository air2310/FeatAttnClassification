function chunk_features = get_features(chunkeeg, sets)
% get_features: Get eeg features for decoding
    %   Inputs:
     %       chunkeeg - EEG data for each different sliding window size
    %        set -Structure describing how the decoding analysis will run.
    %   Outputs:
    %       chunkfeatures - Frequency transformed EEG data for each different sliding window size

    %% Extract features
    % Preallocate
    chunk_features = cell(sets.n.chunksizes,sets.n.hzstates);
    
    % Loop through sliding windows
    for ii_chunk = 1:sets.n.chunksizes
        % Exract data
        DATA = chunkeeg{ii_chunk};
        
        % Run FFT
        n_x = sets.epoch_chunk{ii_chunk}.n.x;
        tmp = abs( fft( DATA) )/n_x;
        tmp(2:end-1,:,:) = tmp(2:end-1,:,:)*2;
        
        % Get indices of meaningful frequencies
        idx_hz = [sets.Hz.f1_idx{ii_chunk} sets.Hz.f2_idx{ii_chunk} sets.Hz.alpha_idx{ii_chunk}(1:end-1)];
        hzcond_idx.Basic = [1 2];
        hzcond_idx.Harmonic = [1 2 3 4];
        hzcond_idx.Alpha = [1 2 5:length(idx_hz)];
        hzcond_idx.AlphaPlusHarmonic = 1:length(idx_hz);
        
        % Extract meaningful frequencies
        baselinefreqs = tmp(idx_hz, :, :);
        
        % get feature vectors
        for ii_hzstate = 1:sets.n.hzstates
            dat = baselinefreqs(hzcond_idx.(sets.str.HzState{ii_hzstate}), :, :);
            
            tmp = dat(1,:,:);
            for HH = 2:size(dat,1)
                tmp = cat(2, tmp, dat(HH,:,:));
            end
            chunk_features{ii_chunk, ii_hzstate} = squeeze(tmp);
        end
    end
end