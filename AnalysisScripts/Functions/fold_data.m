function [features, labels, idx_train, idx_test] = fold_data(ii_fold, candidate_feats, candidate_labels, partition)
% Get features and labels for each fold
% Partition data
if isstring(partition)
    idx_train = 1:length(candidate_labels.train);
    idx_test = 1:length(candidate_labels.test);
else
%     idx_train_tmp = training(partition,ii_fold);
%     idx_test_tmp = test(partition,ii_fold);
%     
%     % partition based on trials
%     tmp = unique(candidate_labels.train_trial);
%     idx_train = ismember(candidate_labels.train_trial,tmp(idx_train_tmp));
%     idx_test = ismember(candidate_labels.test_trial,tmp(idx_test_tmp));
    
    % Get partition
    idx_train_tmp = training(partition,ii_fold);
    idx_test_tmp = test(partition,ii_fold);
    
    % Find trials of each label type
    trials = unique(candidate_labels.train_trial);
    triallabels = NaN(length(trials),1);
    for ii = 1:length(trials)
        triallabels(ii) = mean(candidate_labels.train(candidate_labels.train_trial == trials(ii)));
    end

    % Partition them
    tmp1 = trials(triallabels==1); tmp2 = trials(triallabels==2);
    idx_train = ismember(candidate_labels.train_trial,[tmp1(idx_train_tmp) tmp2(idx_train_tmp)]);
    idx_test = ismember(candidate_labels.test_trial,[tmp1(idx_test_tmp) tmp2(idx_test_tmp)]);

end

% labels
labels.train = candidate_labels.train(idx_train)';
labels.test = candidate_labels.test(idx_test)';

% features
features.train = candidate_feats.train(:,idx_train);
features.test = candidate_feats.test(:,idx_test );
end