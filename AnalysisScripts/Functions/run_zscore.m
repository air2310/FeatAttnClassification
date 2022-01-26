function y_predict = run_zscore(features)
% run_zscore
%   Inputs:
%       features - Frequency transformed EEG data for each different sliding window size
%       labels - labels to correspond to the feature data.
%       sets - structure of metadata and settings

% get channel average data for each frequency. 
f_train = cat(1, mean(features.train(1:5, :),1),mean(features.train(6:10, :),1));
f_test = cat(1, mean(features.test(1:5, :),1),mean(features.test(6:10, :),1));

% Get mean and standard deviation. 
MEAN = mean(f_train,2);
STD = std(f_train,[],2);

% Zscore test data
f_test_zscored = (f_test-MEAN)./STD;
[~,y_predict] = max(f_test_zscored);


