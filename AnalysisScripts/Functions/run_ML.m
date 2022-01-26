function run_ML(chunk_features, chunklabels, sets, runopts, decodestring)
% run_ML - run various machine learning classifiers. 
%   Inputs:
%       chunk_features - Frequency transformed EEG data for each different sliding window size
%       chunklabels - labels to correspond to the feature data. 
%       sets - structure of metadata and settings
%       runopts - run options for this decoding run
%       decodestring - which decoder to run. Options: MLP, LDA, KNN,
%       zscore, SVM
%       LR?,RF?



%% Run!
disp(['Running: ' decodestring])

% Preallocate 
for ii_hzstate = 1:sets.n.hzstates
    ACCURACY.(sets.str.HzState{ii_hzstate}) = cell(sets.n.cols, sets.n.chunksizes);
end
ACCMEAN = NaN(sets.n.cols,  sets.n.chunksizes, sets.n.hzstates);

%Loop through chunk sizes and frequency conditions.
for ii_chunk = 1:sets.n.chunksizes % Sliding window size. 
    for ii_hzstate = 1:sets.n.hzstates % combinations of features to use
        for ii_col = 1:sets.n.cols % Colour couterbalancing condition
            % Display where we're up to
            disp(['Sliding window size: ' num2str(sets.timing.secs.chunksizes(ii_chunk)) ' s. Features: ' sets.str.HzState{ii_hzstate} '. Colour cond: ' num2str(ii_col)])

            % Whittle down data for the specific run options we've set.
            [candidate_feats, candidate_labels] = set_runopts(chunk_features{ii_chunk, ii_hzstate}, chunklabels{ii_chunk}, ii_col, runopts);
            
            
            
            % Partition data
            if runopts.testdat ~= runopts.traindat
                partition = "all";
                nfold = 1;
                n.epochs = length(candidate_labels.test);
            else
                n.epochs = length(candidate_labels.train);
                partition = cvpartition(n.epochs,'KFold',sets.n.folds);
                nfold = sets.n.folds;
            end
            
            % preallocate
            ACCURACY.(sets.str.HzState{ii_hzstate}){ii_col, ii_chunk} = NaN(n.epochs,1);
            
            % Fold data
            for ii_fold = 1:nfold
                % Get training and testing indices
                [features, labels, ~, idx_test] = fold_data(ii_fold, candidate_feats, candidate_labels, partition);
%                 disp( [ sum(labels.train==1) sum(labels.train==2)])

                % Run Classifier
                if strcmp(decodestring, 'MLP')
                    y_predict = run_MLP(features, labels, sets);
                    
                elseif strcmp(decodestring, 'LDA')
                    classifier = fitcdiscr( features.train', labels.train);%,'OptimizeHyperparameters', 'auto');
                    y_predict = predict( classifier, features.test' )';
                    
                elseif strcmp(decodestring, 'KNN')
                    classifier = fitcknn( features.train', labels.train);%,'Distance', 'spearman', 'OptimizeHyperparameters', {'NumNeighbors'}); 
                    y_predict = predict( classifier, features.test' )';
                    
                elseif strcmp(decodestring, 'SVM')
                    classifier = fitcsvm(features.train', labels.train, 'KernelFunction','rbf','KernelScale','auto');
                    
                    y_predict = predict( classifier, features.test' )';
                    
                elseif strcmp(decodestring, 'LR_L1')
                    classifier = fitclinear(features.train', labels.train,...
                        'Learner','logistic','Solver','sgd','Regularization','lasso',...
                        'Lambda','auto');

                    y_predict = predict( classifier, features.test' )';
                    
                 elseif strcmp(decodestring, 'LR_L2')
                    classifier = fitclinear(features.train', labels.train,...
                        'Learner','logistic','Solver','sgd','Regularization','ridge',...
                        'Lambda','auto');
                    
                    y_predict = predict( classifier, features.test' )';

                elseif strcmp(decodestring, 'zscore')
                    if ii_hzstate == 1
                        y_predict = run_zscore(features);
                    else
                        y_predict = NaN(size(labels.test));
                    end
                end
                
                % Get Accuracy
                acc = round(y_predict) == labels.test;
%                 sum(acc)/length(acc)
                ACCURACY.(sets.str.HzState{ii_hzstate}){ii_col, ii_chunk}(idx_test) = acc;
            end
            
            % Store mean accuracy for conditions. 
            ACCMEAN(ii_col, ii_chunk, ii_hzstate) = 100*sum( ACCURACY.(sets.str.HzState{ii_hzstate}){ii_col, ii_chunk})/n.epochs;
            disp(['Accuracy = ' num2str(ACCMEAN(ii_col, ii_chunk, ii_hzstate))])
            
            % Reorganise ACCURACY
            tmp = NaN(max(candidate_labels.test_trial), max(candidate_labels.test_chunk));
            for ii = 1:length(candidate_labels.test)
               tmp(candidate_labels.test_trial(ii), candidate_labels.test_chunk(ii)) = ACCURACY.(sets.str.HzState{ii_hzstate}){ii_col, ii_chunk}(ii);
            end
            tmp(isnan(nanmean(tmp, 2)),:) = [];
            ACCURACY.(sets.str.HzState{ii_hzstate}){ii_col, ii_chunk} = tmp;
        end
    end
end

%% Plot results
plot_subresults(ACCMEAN, sets, decodestring, runopts)

%% Save output
trainstring = ['Train' sets.str.testtrainopts{runopts.traindat} 'Test' sets.str.testtrainopts{runopts.testdat} sets.str.excludemotepochs{runopts.excludemotepochs}];
save([sets.direct.resultssub 'ACCURACY_' decodestring '_' trainstring '.mat'], 'ACCURACY', 'ACCMEAN')

end