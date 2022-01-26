function ACCMEAN_ALL = collate_MLACC(sets, runopts, decodestring)
% collate_MLACC: collate the accuracy scores for the machine learning
% classifier listed
%   Inputs:
%       sets - structure of metadata and settings
%       runopts - run options for this decoding run
%       decodestring - which decoder to run. Options: MLP, LDA, KNN,
%   Outputs:
%       ACCMEAN_ALL - structure containing accuracy across all subjects and
%       training conditions for input classifier.

%% Preallocate
disp(['collating data: ' decodestring])
ACCMEAN_ALL = NaN(sets.n.cols, sets.n.chunksizes, sets.n.hzstates, sets.n.subs);
 
%% Loop through subjects to get data.

subcount = 0;
for SUB = 1:sets.n.sub_ids
    %% Subject settings
    % Exclude excluded participants
    if ismember(SUB, [7 19])
        continue
    end
    
    % count up subjects
    subcount = subcount+1;
    
    % Subject strings and directoris.
    runopts.subject = SUB;
    sets = setup_subject_directories(sets, runopts);
    
    %% Load data
    trainstring = ['Train' sets.str.testtrainopts{runopts.traindat} 'Test' sets.str.testtrainopts{runopts.testdat} sets.str.excludemotepochs{runopts.excludemotepochs}];
    load([sets.direct.resultssub 'ACCURACY_' decodestring '_' trainstring '.mat'], 'ACCURACY', 'ACCMEAN')
    
    %% Account for z-scores (no extra conditions)
    if strcmp(decodestring, 'zscore')
        ACCMEAN(:,:,2:end) = NaN;
    end
    
    %% Collate
    ACCMEAN_ALL(:,:,:,subcount) = ACCMEAN;
end

%% Save out results
% save results and things we need to plot them. 
save([sets.direct.results_group 'ACCURACY_' decodestring '_' trainstring '.mat'], 'ACCMEAN_ALL')
end