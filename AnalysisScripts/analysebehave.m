clear 
clc
close all

%% directories

for SUB = 30
    
str.sub = ['S' num2str(SUB)];

direct.toolbox = '..\..\toolbox\';
addpath([direct.toolbox 'Colormaps\'])

direct.data = '..\data\ORIG\BEHAVE\';
direct.results = ['..\results\' str.sub '\'];
if ~exist(direct.results, 'dir'); mkdir(direct.results); end

    
FNAME = dir([direct.data str.sub '*.mat']);
load([direct.data FNAME.name], 'RESPONSE', 'directions', 'n', 'TRIAL_TABLE', 's', 'mon', 'str') ;

str.attnstate = {'multifreq' 'singlefreq'};
str.sub = ['S' num2str(SUB)];

%% Responses - sort out

figure; plot(RESPONSE)

dat = [zeros(1,n.trials); diff(RESPONSE)];
response = [];
for TRIAL = 1:n.trials
    idx = find(dat(:,TRIAL)>0);
    
    for ii = 1:length(idx)
        response = [response; RESPONSE(idx(ii), TRIAL) TRIAL idx(ii) NaN];
    end
end

%% Score


responseperiod = [0.1 s.postmovebreak]*mon.ref;

responseopts.miss = 0;
responseopts.incorrect = 1;
responseopts.falsealarm = 2; % more than one response
responseopts.correct = 3;

ACCURACY = NaN(n.dirChangesTotal,1);
RT = NaN(n.dirChangesTotal,1);
COND = NaN(n.dirChangesTotal,2); % Attentioncond, colour

movements = 0;
for TRIAL = 1:n.trials
    
    % correct answer
    correct = TRIAL_TABLE.movedir__Attd_UnAttd{TRIAL,1};
    for DD = 1:4
       correct(correct==directions(DD)) = DD ;
    end
    
    %answer period
    tmp = TRIAL_TABLE.moveframe__Attd_UnAttd{TRIAL,1};
    moveframes = NaN(n.dirChangesTrial,2);
    for ii = 1:2
        moveframes(:,ii) = tmp + responseperiod(ii);
    end
    
    %% get accuracy
    for ii = 1:n.dirChangesTrial
         movements = movements + 1;
         % set conditions
         COND(movements,1) = TRIAL_TABLE.ATTENTIONCOND(TRIAL);
         COND(movements,2) = TRIAL_TABLE.Col__Attd_UnAttd(TRIAL,1);
         
         % get eligible responses
         
         eligibleresponses = response(response(:,2)==TRIAL,:);
         
         idx = find(response(:,2)==TRIAL & response(:,3) > moveframes(ii,1) & response(:,3) < moveframes(ii,2));
         
         % fill accuracy data
         if isempty(idx)
             ACCURACY(movements) = responseopts.miss;
             
         elseif length(idx) >1
             ACCURACY(movements) = responseopts.falsealarm;
             
         elseif response(idx,1) == correct(ii)
             ACCURACY(movements) = responseopts.correct;
             RT(movements) = response(idx,3) -  TRIAL_TABLE.moveframe__Attd_UnAttd{TRIAL,1}(ii);
         else
             ACCURACY(movements) = responseopts.incorrect;
             RT(movements) = response(idx,3) -  TRIAL_TABLE.moveframe__Attd_UnAttd{TRIAL,1}(ii);
         end
    end
end

%% Plot accuracy overall

MACC = 100*[sum(ACCURACY==responseopts.correct) sum(ACCURACY==responseopts.incorrect) sum(ACCURACY==responseopts.miss) sum(ACCURACY==responseopts.falsealarm)]./n.dirChangesTotal;

h = figure;
pie(MACC)
str.accuracy = {'correct' 'incorrect' 'miss' 'false alarm'};
legend(str.accuracy)
colormap(viridis)

tit = [str.SUB ' Grand Avg. Accuracy'];
title(tit)
saveas(h, [direct.results tit '.png'])

%% Accuracy by Attention condition

dat1 = ACCURACY(COND(:,1)==1);
dat2 = ACCURACY(COND(:,1)==2);
n.dirchangescond = length(dat1);

MACC1 = 100*[sum(dat1==responseopts.correct) sum(dat1==responseopts.incorrect) sum(dat1==responseopts.miss) sum(dat1==responseopts.falsealarm)]./n.dirchangescond;
MACC2 = 100*[sum(dat2==responseopts.correct) sum(dat2==responseopts.incorrect) sum(dat2==responseopts.miss) sum(dat2==responseopts.falsealarm)]./n.dirchangescond;

BEHAVE.ACC_By_Attn = [MACC1; MACC2];

h = figure;

bar([MACC1; MACC2]')
colormap([0 0 1; 0 1 1]);

set(gca, 'xticklabel', {'correct' 'incorrect' 'miss' 'false alarm'}, 'tickdir', 'out')
box('off')
xlabel('Response Type')
ylabel('Percentage of responses (%)');
legend(str.attnstate)

tit = [str.sub ' Acc by Attention state'];
title(tit)
saveas(h, [direct.results tit '.png'])


%% Accuracy by Colour

dat1 = ACCURACY(COND(:,2)==1);
dat2 = ACCURACY(COND(:,2)==2);
n.dirchangescond = length(dat1);

MACC1 = 100*[sum(dat1==responseopts.correct) sum(dat1==responseopts.incorrect) sum(dat1==responseopts.miss) sum(dat1==responseopts.falsealarm)]./n.dirchangescond;
MACC2 = 100*[sum(dat2==responseopts.correct) sum(dat2==responseopts.incorrect) sum(dat2==responseopts.miss) sum(dat2==responseopts.falsealarm)]./n.dirchangescond;

BEHAVE.ACC_By_Col = [MACC1; MACC2];

h = figure;

bar([MACC1; MACC2]')
colormap([0 0 0; 1 1 1]);

set(gca, 'xticklabel', {'correct' 'incorrect' 'miss' 'false alarm'}, 'tickdir', 'out')
box('off')
xlabel('Response Type')
ylabel('Percentage of responses (%)');
legend(str.cue)

tit = [str.sub ' Acc by Colour'];
title(tit)
saveas(h, [direct.results tit '.png'])

%% Reaction time data

% transform from frames to seconds
RT2 = RT./mon.ref;

% Attention state
dat1 = RT2(COND(:,1)==1 & ACCURACY==responseopts.correct);
dat2 = RT2(COND(:,1)==2 & ACCURACY==responseopts.correct);

M_attn = [mean(dat1) mean(dat2)];
E_attn = [std(dat1)/sqrt(length(dat1)) std(dat2)/sqrt(length(dat2))];

% Colour
dat1 = RT2(COND(:,2)==1 & ACCURACY==responseopts.correct);
dat2 = RT2(COND(:,2)==2 & ACCURACY==responseopts.correct);

M_col = [mean(dat1) mean(dat2)];
E_col = [std(dat1)/sqrt(length(dat1)) std(dat2)/sqrt(length(dat2))];

BEHAVE.RT_By_AttnandCol = [M_attn; M_col];

LIMIT = [0, max([M_attn M_col ]) + max([E_attn E_col ]) + 0.1];
h = figure;
subplot(1,2,1)
barwitherr(E_attn, M_attn);
set(gca, 'xticklabel', str.attnstate, 'tickdir', 'out')
box('off')
xlabel('Attention State')
ylabel('RT (s)')
xlim([ 0 3])
ylim(LIMIT)
title('RT by Attn. state')

subplot(1,2,2)
barwitherr(E_col, M_col);
set(gca, 'xticklabel', str.cue, 'tickdir', 'out')
box('off')
xlabel('Colour Attended')
ylabel('RT (s)')
xlim([ 0 3])
ylim(LIMIT)
title('RT by colour')

tit = [str.SUB ' Reaction Time Data'];
suptitle(tit)
saveas(h, [direct.results tit '.png'])

%% Save

BEHAVE.ACCURACY = ACCURACY;
BEHAVE.RT = RT2;

save([direct.results str.sub 'BehaveResults.mat'], 'BEHAVE', 'str')
end