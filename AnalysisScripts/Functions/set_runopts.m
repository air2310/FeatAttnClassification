function [features, labels] = set_runopts(features_tmp, labels_tmp, ii_col, runopts)
%% Get features and labels specific to the settings we've set for this run
% set which non-noisy epochs we'll keep.
keep.noise = ~(labels_tmp.reject_motionepoch | labels_tmp.reject_noise);
switch ii_col
    case 1
        keep.col = labels_tmp.colcued == labels_tmp.hzcued;
    case 2
        keep.col = labels_tmp.colcued ~= labels_tmp.hzcued;
end

% Get training labels and featues.
keep.stimstate = labels_tmp.stimstate==runopts.traindat;
labels.train = labels_tmp.hzcued(keep.stimstate & keep.noise & keep.col);
features.train = features_tmp(:,keep.stimstate & keep.noise & keep.col);

% Get testing labels and featues.
keep.stimstate = labels_tmp.stimstate==runopts.testdat;
labels.test = labels_tmp.hzcued(keep.stimstate & keep.noise & keep.col);
labels.test_trial = labels_tmp.trial(keep.stimstate & keep.noise & keep.col);
labels.test_chunk = labels_tmp.chunk(keep.stimstate & keep.noise & keep.col);
features.test = features_tmp(:,keep.stimstate & keep.noise & keep.col);

% Even out numbers
if sum(labels.train==1) ~= sum(labels.train==2)
    % Even this out. 
    dat = [ sum(labels.train==1) sum(labels.train==2)];
    [longer, longer_idx] = max(dat);
    [shorter, ~] = min(dat);
    
    tmp = randsample(find(labels.train==longer_idx), longer-shorter);
    labels.train(tmp) = [];
    features.train(:,tmp) = [];
   
end

if sum(labels.test==1) ~= sum(labels.test==2)
    % Even this out. 
    dat = [ sum(labels.test==1) sum(labels.test==2)];
    [longer, longer_idx] = max(dat);
    [shorter, ~] = min(dat);
    
    tmp = randsample(find(labels.test==longer_idx), longer-shorter);
    labels.test(tmp) = [];
    labels.test_trial(tmp) = [];
    labels.test_chunk(tmp) = [];
    features.test(:,tmp) = [];
   
end

end