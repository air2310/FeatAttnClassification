function [sets] = setup_subject_directories(sets, runopts)
% setup_subject_directories: update directory paths for specific subjects
%   Inputs:
%        set - Structure describing how the decoding analysis will run.
%        runopts - Structure describing decoding run options.
%
%   Outputs:
%        set.direct - updated to include subject specific data and
%               results paths. 
%        set.str - updated in include subject strings

%% Get out origional data
direct = sets.direct;
str = sets.str;

%% Subject strings
str.sub = ['S' num2str(runopts.subject)];

if runopts.subject <10
    str.sub_bids = ['sub-0' num2str(runopts.subject)];
else
    str.sub_bids = ['sub-' num2str(runopts.subject)];
end

%% Data directories. 
direct.datasub.eeg = [sets.direct.data str.sub_bids '\eeg\'];
direct.datasub.behave = [sets.direct.data 'sourcedata\' str.sub_bids '\behave\'];

%% Data filenames
bids.taskname = 'FeatAttnDec';
bids.filename = [str.sub_bids '_task-' bids.taskname ];

% EEG
direct.filename.eeg = [direct.datasub.eeg bids.filename '_eeg.eeg'];

% Behave
direct.filename.behave = [direct.datasub.behave bids.filename '_behav.mat'];       
        
%% Results directories
direct.resultssub = [sets.direct.results str.sub '\'];
if ~exist(direct.resultssub, 'dir'); mkdir(direct.resultssub); end

% For loading old cca templates
direct.resultssub_backup = [sets.direct.main 'Results\' str.sub  '\'];

%% Data filenames
bids.taskname = 'FeatAttnDec';
bids.filename = [str.sub_bids '_task-' bids.taskname ];

direct.filename.trialeeg = [direct.resultssub bids.filename '_trialeeg.mat'];

%% Save back to settings
sets.direct = direct;
sets.str = str;

end