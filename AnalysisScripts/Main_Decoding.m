%% Main Script for comparison of multiple methods of Real-time, singletrial, attentional selectivity calculation.
% Created by Angela I. Renton on 25/11/21. Email: angie.renton23@gmail.com

%% start with a clean slate
clear
clc
close all
addpath('Functions/');

% set random seed.
rng('default') % For reproducibility

%% Set decoding run options
% Decoding to run
runopts.zscore =1; % Z-score
runopts.LDA = 1; % Linear discriminant Analysis
runopts.KNN =1; % K-Nearest Neighbours
runopts.MLP = 1; % Multi layer perceptron
runopts.SVM = 1; % Support vector machine with RBF kernel
runopts.LR_L1 = 0; % Logistic regression w/ lasso regularization
runopts.LR_L2 = 1; % Logistic regression w/ ridge regularization

% Data to train and test on
runopts.traindat =2; % 1 = 'Distractor Present (multifreq)', 2 = 'Distractor Absent (singlefreq)'
runopts.testdat = 1; % 1 = 'Distractor Present (multifreq)', 2 = 'Distractor Absent (singlefreq)'

% run individuals vs. collate group
runopts.individuals = 1; % Run the analyses on individual participants
runopts.collate = 1; % Collate results across participants

% Exclude motion epochs
runopts.excludemotepochs = 1; % 1 = exclude , 2 = include (epochs around motion epochs from training and testing) 

%% Generic metadata
sets = setup_metadata();

%% Loop through subjects
if runopts.individuals
        
    for SUB = 1:sets.n.sub_ids
        %% Subject settings
        % Exclude excluded participants
        %(technical issues with recording
        % sessions)
        if ismember(SUB, [7 19])
            continue
        end
        
        % Setup strings and directories for this participant.
        runopts.subject = SUB;
        disp(['Running subject :' num2str(runopts.subject)])
        sets = setup_subject_directories(sets, runopts);
        
        %% Load and organise EEG data
        disp('Loading EEG data')
        trialeeg = get_eeg(sets);
       
        %% Sliding window data extraction
        disp('Extracting sliding window chunks')
        [chunkeeg, chunklabels] = get_slidingwindoweeg(trialeeg,sets);
            
        %% Get features (frequency transformed data).
        disp('Calculating features')
        chunk_features = get_features(chunkeeg, sets);
        
        %% Run decoding
        for ii_method = 1:sets.n.methods
            decodestring = sets.str.methods{ii_method};
            if runopts.(decodestring)
                run_ML(chunk_features, chunklabels, sets, runopts, decodestring)
            end
        end        
    end
end

%% Collate results across subjects
if runopts.collate
    % Collate for each method
    for ii_method = 1:sets.n.methods
        decodestring = sets.str.methods{ii_method};
        if runopts.(decodestring)
            % Get individual subject data
            ACCMEAN_ALL = collate_MLACC(sets, runopts, decodestring);
            
            % Plot data
            plot_groupresults(ACCMEAN_ALL, sets, runopts, decodestring)
        end
    end
end