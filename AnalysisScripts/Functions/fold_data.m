function [features, labels, idx_train, idx_test] = fold_data(ii_fold, candidate_feats, candidate_labels, partition)
% Get features and labels for each fold
% Partition data
if isstring(partition)
    idx_train = 1:length(candidate_labels.train);
    idx_test = 1:length(candidate_labels.test);
else
    idx_train = training(partition,ii_fold);
    idx_test = test(partition,ii_fold);
end

% labels
labels.train = candidate_labels.train(idx_train)';
labels.test = candidate_labels.test(idx_test )';

% features
features.train = candidate_feats.train(:,idx_train);
features.test = candidate_feats.test(:,idx_test );
end