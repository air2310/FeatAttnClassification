function y_predict = run_MLP(features, labels, sets)
% run_MLP
%   Inputs:
%       chunk_features - Frequency transformed EEG data for each different sliding window size
%       chunklabels - labels to correspond to the feature data.
%       sets - structure of metadata and settings
%       runopts - run options for this decoding run

%% Run! 
%Set up patternnet
net = patternnet(sets.netset.hiddenLayerSize, sets.netset.trainFcn, sets.netset.performFcn);
net = init(net);
net.divideFcn = sets.netset.divideFcn;
net.divideMode = sets.netset.divideMode;
net.divideParam = sets.netset.divideParam;

%## Train the Network
[net,tr] = train(net,features.train,labels.train);

% Use the network to predict outputs
y_predict = net(features.test);

end