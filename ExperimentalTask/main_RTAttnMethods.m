clear
clc
close all

%% Attention Neurofeedback Methods
% This is the main script in an experiment designed to figure out how to calculate feedback about participants attentional state
%
% Angela Renton
% 05/11/18


%% Subject

SUB = 0;
str.SUB = ['S' num2str(SUB)];

%% Directories

direct.functions = 'Functions\'; addpath(direct.functions);

direct.toolbox = 'C:\EXPERIMENTS\toolboxes\';

direct.psychtoolbox = [direct.toolbox 'Psychtoolbox\'];
direct.io = [direct.toolbox 'io64'];

addpath(direct.io)

%% Settings

% Options
options.hideCursor = false;
options.flickertype = 2; % Colour Vs Luminance
options.arrow = 0;
options.trigger = 1;

% Monitor Settings
mon.use = 1;
mon.ref = 120;

% Trial settings
n.cond = 2;

n.trials = 240;
n.trialsCond = n.trials/n.cond;

n.blocks = 12;
n.trialsBlock = n.trials/n.blocks;
n.dirs = 4;

% Timing settings
s.trial = 15;
s.cue = 2;
s.Motionblock = 2; % 2 second blocks alternating motion/ no motion = 1 instance of motion every three seconds.
s.motionchangetrial = 4;
s.coherentMotion = 0.5;%0.5;
s.feedback = 0.5;
s.feedbacklatency = 0.5;

s.trialonsetbreak = 1;
s.postmovebreak = 1.5;
s.wiggleroom = 0.5;

tmp = fields(s);
for ii = 1:length(tmp)
    f.(tmp{ii}) = s.(tmp{ii})*mon.ref;
end

% Frequencies
Hz = [6 7.5 ];
n.Hz = length(Hz);
n.cols = 2;

% Directions
directions = [0 90 180 270];

% Keys
key.esc = 27;
key.enter = 13;
key.left = 37;
keyp.up = 38;
key.right = 39;
key.down = 40;

key.response = [key.right key.down key.left keyp.up];

% Colours
colour.WHITE = uint8([1 1 1]*255)';
colour.BLACK =  uint8([0 0 0]*255)';
colour.grey =  uint8([1.5 0.5 0.5]*255)';
colour.red = uint8([1 0 0]*255)';
colour.green = uint8([0 1 0]*255)';
colour.blue = uint8([0 0 1]*255)';

baseline = 127.5;

str.cue = {'BLACK' 'WHITE'};

%% Triggers
n.attnstates = 2;
% Cue Start
% Hz Attend, Col Attend, Training Cond
trig.cue = NaN(n.Hz,n.cols,n.attnstates);

trig.cue(1,:,:) = [
    1 2 ;
    3 4    ];

trig.cue(2,:,:) = [
    5 6;
    7 8    ];

% Trial Start
% Hz Attend, Col Attend, Training Cond
trig.trial = trig.cue + 20;

% Motion Onset
% Hz Attd move, Col move, # of  dot fields, motion direction

trig.motion = NaN(n.Hz,n.cols,n.attnstates, n.dirs);

trig.motion(1,1,:,:) = [
    101 102 103 104;
    105 106 107 108
    ];

trig.motion(1,2,:,:) = trig.motion(1,1,:,:)+8;
trig.motion(2,1,:,:) = trig.motion(1,2,:,:)+8;
trig.motion(2,2,:,:) = trig.motion(2,1,:,:)+8;

% Feedback start
trig.feedback = 222;

% start and stop recording
trig.stopRecording = 254;
trig.startRecording = 255;

%% setup triggering

if options.trigger
    PCuse = 4;
    trig.ioObj = io64;
    trig.status = io64(trig.ioObj);
    
    switch PCuse
        case 1
            trig.options.port = { 'D010'  'D030' };
        case 2
            trig.options.port = { 'D050'  'D050' };
        case 3
            trig.options.port = { '21'  '21' };
        case 4
            trig.options.port = { '21'  '2FF8' };
    end
    n.ports = length(trig.options.port);
    trig.address = NaN(n.ports,1);
    
    for AA = 1:n.ports
        trig.address(AA) = hex2dec( trig.options.port{AA} );
        io64(trig.ioObj, trig.address(AA), 0);
    end
    
    io64(trig.ioObj, trig.address(1), trig.startRecording);
    
end

for ii = 0:255
    io64(trig.ioObj, trig.address(2), ii);
    % a=io64(trig.ioObj, trig.address(2))
    pause(0.05)
    io64(trig.ioObj, trig.address(2), 0);
    pause(0.05)
end
io64(trig.ioObj, trig.address(1), trig.stopRecording);
%% Setup Dots

dots.speed   = 100;%80    % dot speed (pixels/sec)
dots.speed_frame = dots.speed / mon.ref;  % dot speed (pixels/frame)

dots.n       = 200; % number of dots

dots.max_d       = 350;%310;   % maximum radius of  annulus (pixels)
dots.width       = 18;%30; %pixels - larger dots is slower!

n.fields = 4;

dots.order = NaN(n.trials,dots.n*n.Hz);
dots.order_signal = NaN(n.trials,dots.n*n.Hz);
for TT = 1:n.trials
    dots.order(TT,:) = randperm(dots.n*n.Hz);
    dots.order_signal(TT,:) = randperm(dots.n*n.Hz);
end

DOT_XY = NaN(2,dots.n,n.fields);
for FIELD = 1:n.fields
    DOT_XY(:,:,FIELD) = SetupDotCoords(dots)';
end


%% Flicker
t = 0 : 1/mon.ref : s.trial - 1/mon.ref;

FLICKER = NaN(f.trial, dots.n , n.Hz, n.cols);

baselinecol= [0 baseline];

for HH = 1 : n.Hz
    sig = 0.5 + 0.5*sin(2*pi*Hz(HH)*t ); %+ 2*pi*rand);
    
    sig = round(sig);
    sig = sig.*255;
    
    for CC = 1:2
        FLICKER(:,:,HH, CC) = uint8(repmat(sig', 1, dots.n)).*0.5 + baselinecol(CC);
    end
end


%% Direction changes

n.dirs = length(directions);
n.dirChangesTrial = 5;
n.dirChangesTotal = n.dirChangesTrial*n.trials;
n.dirChangesPerDir = n.dirChangesTotal/n.dirs;

% The actual direction changes
dirchanges = [];
for ii = 1:n.dirs
    dirchanges = [dirchanges; ones(n.dirChangesPerDir,1).*directions(ii)];
end
dirchanges = [dirchanges(randperm(n.dirChangesTotal)) dirchanges(randperm(n.dirChangesTotal))];

% Create orderly dirchanges by trial matrix
dirchanges_trial = NaN(n.dirChangesTrial, n.trials, n.cols);
dirchanges_trial(:,:,1) = reshape(dirchanges(:,1), [n.dirChangesTrial, n.trials]) ;
dirchanges_trial(:,:,2) = reshape(dirchanges(:,2), [n.dirChangesTrial, n.trials]) ;

% Decide which frames will have coherent motion.
coherentMotion = zeros(f.trial, n.trials, n.cols);
coherentMotionAngle = NaN(f.trial, n.trials, n.cols);
coherentMotionFrame = NaN(n.dirChangesTrial,  n.trials, n.cols);


for TT = 1:n.trials
    for CC = 1:n.cols
        
        startFrame = f.trialonsetbreak +1;
        for ii = 1:n.dirChangesTrial
            
            remainingBreakFrames = (f.coherentMotion + f.postmovebreak  + 1)*(n.dirChangesTrial - ii+1) +(f.wiggleroom*(n.dirChangesTrial -ii)); % keep some space for the breaks after the remaining motion episodes
            stopFrame = f.trial - remainingBreakFrames - f.coherentMotion; % what is the last possible frame this motion episode could occur on?
            
            
            MotionFrame = randsample(stopFrame - startFrame,1) + startFrame; % Pick a random frame in this bracket
            
            coherentMotion(MotionFrame+1:MotionFrame + f.coherentMotion,TT, CC) = 1; % allocate the motion
            coherentMotionAngle(MotionFrame+1:MotionFrame + f.coherentMotion,TT, CC) = dirchanges_trial(ii,TT,CC);
            coherentMotionFrame(ii, TT, CC) = MotionFrame;
            
            startFrame = MotionFrame + f.coherentMotion + f.postmovebreak ; % update the start of the next motion episode
        end
    end
end


%% Setup Trial DATA

% The variables we'll create:
D = {'BLOCK' 'TRIAL' 'ATTENTIONCOND' 'Col__Attd_UnAttd' 'Freq__Attd_UnAttd' 'Freq__ColA_ColB' 'movedir__Attd_UnAttd' 'movedir__ColA_ColB' 'moveframe__Attd_UnAttd' 'moveframe__ColA_ColB' };
Descriptions = {
    'Block in Exp'
    'Trial in Exp'
    'Presence or absence of distractor freq Cond: 1 - distractors | 2 - no distractors'
    'Colour attended and unattended: 1 - black | 2 - white'
    ['Freq attended and unattended: 1 - ' num2str(Hz(1)) ' | 2 - ' num2str(Hz(2)) ]
    ['Freq by colour, column 1 black, column 2 white: 1 - ' num2str(Hz(1)) ' | 2 - ' num2str(Hz(2)) ]
    'movement dirs (angle in degrees) by attention'
    'movement dirs (angle in degrees) by colour'
    'movement onset frames (angle in degrees) by attention'
    'movement onset frames (angle in degrees) by colour'};


% ######### Blocks - 'BLOCK' #########
DATA.BLOCK = [];
for ii = 1:n.blocks
    DATA.BLOCK = [DATA.BLOCK; ones(n.trialsBlock,1).*ii];
end

blockendtrials = n.trialsBlock:n.trialsBlock:n.trials;

% ######### Trials - 'TRIAL' #########
DATA.TRIAL = (1:n.trials)';

% ######### Neurofeedback condition - 'NFCOND' #########
% 1 - congruent | 2 - replay | 3 - incongruent
DATA.ATTENTIONCOND = [];
for ii = 1:n.cond
    DATA.ATTENTIONCOND = [DATA.ATTENTIONCOND; ones(n.trialsCond,1).*ii];
end

DATA.ATTENTIONCOND = DATA.ATTENTIONCOND(randperm(n.trials));

% ######### Colour attended and unattended - 'Col__Attd_UnAttd' #########
% 1 - black | 2 - white

DATA.Col__Attd_UnAttd = NaN(n.trials,2);
for ii = 1:n.cond
    % Index so that each condition is separately counterbalanced
    idx.nfcond = find( DATA.ATTENTIONCOND == ii);
    
    % Equal numbers of attending black and white
    tmp = [ones(n.trialsCond/n.cols,1)*1; ones(n.trialsCond/n.cols,1)*2];
    DATA.Col__Attd_UnAttd(idx.nfcond,1) = tmp(randperm(n.trialsCond));
    
end
DATA.Col__Attd_UnAttd(:,2) = ~(DATA.Col__Attd_UnAttd(:,1) - 1) + 1;

% ######### Frequency attended and unattended - 'Freq__Attd_UnAttd' #########
% 1 - Hz1 | 2 - Hz2

DATA.Freq__Attd_UnAttd = NaN(n.trials,2);
for ii = 1:n.cond
    for col = 1:n.cols
        % Index so that each condition and colour is separately counterbalanced
        idx.nfcond_col = find( DATA.Col__Attd_UnAttd(:,1)==col & DATA.ATTENTIONCOND == ii);
        
        % Equal numbers of black and white attended trials flickering at each frequency
        tmp = [ones(n.trialsCond/(n.cols*n.Hz),1); ones(n.trialsCond/(n.cols*n.Hz),1)*2];
        DATA.Freq__Attd_UnAttd(idx.nfcond_col, 1) = tmp(randperm(n.trialsCond/2));
    end
end

DATA.Freq__Attd_UnAttd(:,2) = ~(DATA.Freq__Attd_UnAttd(:,1) - 1) + 1;

% ######### Frequency for black and white dots - 'Freq__ColA_ColB'  #########
% 1 - Hz1 | 2 - Hz2
% This counterbalancing has already been performed allocating colours and
% frequencies to be attended and unattended equally in each condition. Here
% we just reassign this counterbalancing into a new variable to make life
% easier and clearer down the line.

Freq__ColA_ColB = NaN(n.trials,2);
for freq = 1:2
    for col = 1:n.cols
        for Attn_state = 1:2
            idx.freq_col = DATA.Col__Attd_UnAttd(:,Attn_state)==col & DATA.Freq__Attd_UnAttd(:,Attn_state) == freq;
            DATA.Freq__ColA_ColB( idx.freq_col, col) = freq;
        end
    end
end
% any(abs(diff([DATA.Freq__ColA_ColB]'))~=1) %- should be 0 if correct

% ######### movement directions and grame presented in each trial by attention - 'movedir__Attd_UnAttd' #########
% The N movement directions presented for each colour on each trial,
% shuffled to be arranged by attention insteadd of colour

dirchanges_trial_attn = NaN(size(dirchanges_trial));
coherentMotionFrame_attn = NaN(size(coherentMotionFrame));
for Attn_state = 1:2
    for col = 1:2
        idx.attn_col = find(DATA.Col__Attd_UnAttd(:,Attn_state) == col);
        
        dirchanges_trial_attn(:,idx.attn_col, Attn_state) =  dirchanges_trial(:,idx.attn_col, col) ;
        coherentMotionFrame_attn(:,idx.attn_col, Attn_state) =  coherentMotionFrame(:,idx.attn_col, col) ;
    end
    
end

% ######### movement directions presented in each trial - 'movedir__ColA_ColB' #########
% The N movement directions presented for each colour on each trial
DATA.movedir__ColA_ColB = cell(n.trials,2);
DATA.movedir__Attd_UnAttd = cell(n.trials,2);
DATA.moveframe__ColA_ColB = cell(n.trials,2);
DATA.moveframe__Attd_UnAttd = cell(n.trials,2);

for TT = 1:n.trials
    for col = 1:2
        DATA.movedir__ColA_ColB{TT,col} = dirchanges_trial(:,TT,col)';
        DATA.movedir__Attd_UnAttd{TT,col} = dirchanges_trial_attn(:,TT,col)';
        
        DATA.moveframe__ColA_ColB{TT,col} = coherentMotionFrame(:,TT,col)';
        DATA.moveframe__Attd_UnAttd{TT,col} = coherentMotionFrame_attn(:,TT,col)';
    end
end


%% Allocate data to a big ass table

TRIAL_TABLE = table;
TRIAL_TABLE.Properties.Description = 'Trial by trial metadata';
TRIAL_TABLE.Properties.UserData = str.SUB;
for ii = 1:length(D)
    TRIAL_TABLE.(D{ii}) =  DATA.(D{ii});
    
    TRIAL_TABLE.Properties.VariableDescriptions{ii} = Descriptions{ii};
end

%% This will become an even bigger frame by frame table one day
RESPONSE = zeros(f.trial, n.trials);

% balancefactors
%% Setup Psychtoolbox

% Graphics
AssertOpenGL;

% open the screen

screens=Screen('Screens');
screenNumber = mon.use; %max(screens);

switch options.flickertype
    case 1
        [windowPtr, rect] = Screen('OpenWindow', screenNumber, 0 );
        
    case 2
        [windowPtr, rect] = Screen('OpenWindow', screenNumber, baseline);
end

% Enable alpha blending with proper blend-function.
% We need it for drawing of smoothed points:
Screen('BlendFunction', windowPtr, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

% other screen things

[mon.centre(1), mon.centre(2)] = RectCenter(rect);
fix_r       = 6; % radius of fixation point (deg)
fix_coord = [mon.centre-fix_r mon.centre+fix_r];

% mon.ref=Screen('FrameRate',windowPtr);      % frames per second
mon.ifi=Screen('GetFlipInterval', windowPtr);
if mon.ref==0
    mon.ref=1/mon.ifi;
end

Priority(MaxPriority(windowPtr));
if options.hideCursor
    HideCursor;	% Hide the mouse cursor
end

% Initial flip...
Screen('Flip', windowPtr);

% Fixation cross
dots.fixation = [
    mon.centre(1) - 8, mon.centre(1) - 2;
    mon.centre(2) - 2, mon.centre(2) - 8;
    mon.centre(1) + 8, mon.centre(1) + 2;
    mon.centre(2) + 2, mon.centre(2) + 8];

% Select specific text font, style and size:
Screen('TextFont',windowPtr, 'Arial Black');
Screen('TextSize',windowPtr, 60);
% Screen('TextStyle', windowPtr, 1+2);

%% Initialise loop variables

breaker = false;

n.signaldots = 200;
n.signaldots_half = n.signaldots/2 -5;

vbl = NaN(f.trial, n.trials);

ColCond = [ 1 2 1 2];
ColCond_signal2 = [0 1].*255;

colvect = NaN(3, dots.n, n.fields);

FRAME_cue  =0;
FRAME = 0;
FRAME_feedback = 0;

ACC = NaN(n.dirChangesTrial, n.trials);
RT = NaN(n.dirChangesTrial, n.trials);

%% Present Dots!!

for TRIAL = 1:n.trials
    trigsend = 0; % reset trigger
    
    if TRIAL == 1
        trialstage = 0;
        Screen('TextSize',windowPtr, 40);
        while true
            str.block = ['BLOCK: 1 of ' num2str(n.blocks) ];
            DrawFormattedText(windowPtr,str.block, 'center', mon.centre(2) - 250,colour.BLACK);
            DrawFormattedText(windowPtr,'Press [ENTER]', 'center', mon.centre(2) - 150 ,colour.BLACK);
            
            % flip
            flipper
            
            % break if enter
            [~, ~, keyCode, ~] = KbCheck();
            if any(find(keyCode)==key.enter)
                break;
            end
        end
        Screen('TextSize',windowPtr, 60);
        
    end

    %% Cue
    trialstage = 1;
    
    fake_attention = round(10+ n.signaldots_half + n.signaldots_half*sin(2*pi*0.25*t +rand*2*pi));
    
    for FRAME_cue = 1:f.cue
        DrawFormattedText(windowPtr,str.cue{TRIAL_TABLE.Col__Attd_UnAttd(TRIAL, 1)}, 'center', 'center',colour.(str.cue{TRIAL_TABLE.Col__Attd_UnAttd(TRIAL, 1)}));
        flipper
    end
    
    %% Moving Dots!
    trialstage = 2;
    
    responseframe = - f.feedbacklatency - 1;
    for FRAME = 1:f.trial
        %% update dot position
        %         disp(squeeze(coherentMotionAngle(FRAME,TRIAL, :))')
        for FIELD = 1:n.fields
            
            if ismember(FIELD, [1 2])
                theta_A = linspace(0,360, dots.n); % random dot angles
            else
                % random motion unless otherwise specified
                if isnan(coherentMotionAngle(FRAME,TRIAL, FIELD - 2) ) || TRIAL_TABLE.Col__Attd_UnAttd(TRIAL,2) == FIELD-2
                    theta_A = linspace(0,360, dots.n);
                else
                    theta_A = ones(dots.n,1).* coherentMotionAngle(FRAME,TRIAL, FIELD - 2);
                end
            end
            
            dots.dxdy(:,1,FIELD) = dots.speed_frame*cosd(theta_A);
            dots.dxdy(:,2,FIELD) = dots.speed_frame*sind(theta_A);
            
            % update
            DOT_XY(:,:,FIELD) = DOT_XY(:,:,FIELD) + dots.dxdy(:,:,FIELD)';
            
            % Deal with dots that move off screen
            IDX = find( sqrt( DOT_XY(1,:,FIELD).^2 + DOT_XY(2,:,FIELD).^2 ) >= dots.max_d );
            if ~isempty(IDX)
                ang = atan2d( DOT_XY(1,IDX,FIELD), DOT_XY(2,IDX,FIELD) ) + 180;
                DOT_XY(2,IDX,FIELD) = ( dots.max_d ).*cosd(ang);
                DOT_XY(1,IDX,FIELD) = ( dots.max_d ).*sind(ang);
            end
            
        end
        
        %% update luminance
        for FIELD = 1:n.fields
            
            % Calculate Scaling
            scaler = (dots.max_d - sqrt( DOT_XY(1,:,FIELD).^2 + DOT_XY(2,:,FIELD).^2));
            scaler = 3*scaler/dots.max_d;
            scaler(scaler > 1) = 1;
            
            %get signal
            if FIELD <= n.Hz
                siga = FLICKER(FRAME,:,TRIAL_TABLE.Freq__ColA_ColB(TRIAL, ColCond(FIELD)), ColCond(FIELD));
            else
                siga = repmat(ColCond_signal2(FIELD-2), 1, dots.n); % switch to better way to switch colour
            end
            
            % Execute Scaling
            switch ColCond(FIELD)
                case 1
                    sigb = -((-siga + baseline).*scaler) + baseline;
                case 2
                    sigb = ((siga-baseline).*scaler)+baseline;
            end
            colvect(:,:,FIELD) = repmat(sigb,3,1);
        end
        
        %% Keys for responses and things
        
        [~, ~, keyCode, ~] = KbCheck();
        
        if find(keyCode)==key.esc
            if options.hideCursor
                ShowCursor
            end
            breaker = true;
            break;
        end
        
        if ismember( find(keyCode), key.response)
            RESPONSE(FRAME, TRIAL) = find( ismember(  key.response, find(keyCode))); %right down left up
            responseframe = FRAME;
            responsetmp = RESPONSE(FRAME, TRIAL);
        end
        
        %% Update Fake Attention
        
        Ndotsattended = fake_attention(FRAME);
        
        signaldots{1} = zeros(dots.n,1);
        signaldots{2} = zeros(dots.n,1);
        %         switch TRIAL_TABLE.NFCOND(TRIAL,:)
        %             case 1
        %                 signaldots{TRIAL_TABLE.Col__Attd_UnAttd(TRIAL,1)}(1:Ndotsattended) = 1;
        %                 signaldots{TRIAL_TABLE.Col__Attd_UnAttd(TRIAL,2)}(1:(n.signaldots - Ndotsattended)) = 1;
        %             case 2
        %                 signaldots{TRIAL_TABLE.Col__Attd_UnAttd(TRIAL,1)}(1:(n.signaldots - Ndotsattended)) = 1;
        %                 signaldots{TRIAL_TABLE.Col__Attd_UnAttd(TRIAL,2)}(1:Ndotsattended) = 1;
        %             case 3
        %                 signaldots{TRIAL_TABLE.Col__Attd_UnAttd(TRIAL,1)}(1:(n.signaldots - Ndotsattended)) = 1;
        %                 signaldots{TRIAL_TABLE.Col__Attd_UnAttd(TRIAL,2)}(1:Ndotsattended) = 1;
        %         end
        
        %       No signal dot number variation
        signaldots{TRIAL_TABLE.Col__Attd_UnAttd(TRIAL,1)}(1:100) = 1;
        signaldots{TRIAL_TABLE.Col__Attd_UnAttd(TRIAL,2)}(1:100) = 1;
        
        signaldots_elim =[signaldots{1}' signaldots{2}'];
        signaldots_elim = signaldots_elim(dots.order_signal(TRIAL, :));
        %       signaldots_elim = dots.order_signal(TRIAL,  ~~[signaldots{TRIAL_TABLE.Col__Attd_UnAttd(TRIAL, 1)}' signaldots{TRIAL_TABLE.Col__Attd_UnAttd(TRIAL, 2)}']);
        
        %% Draw dots
        
        switch TRIAL_TABLE.ATTENTIONCOND(TRIAL)
            case 1
                % randomise order of flickering dots
                tmp_xy = [DOT_XY(:,:,1) DOT_XY(:,:,2)]; tmp_xy = tmp_xy(:,dots.order(TRIAL, :));
                tmp_col = [colvect(:,:,1) colvect(:,:,2)];  tmp_col = tmp_col(:,dots.order(TRIAL, :));
                
                % randomise order of signal dots
                tmp_xy_sig = [DOT_XY(:,:,3) DOT_XY(:,:,4)]; tmp_xy_sig = tmp_xy_sig(:,dots.order_signal(TRIAL, :)); tmp_xy_sig = tmp_xy_sig(:,~~signaldots_elim);
                tmp_col_sig = [colvect(:,:,3) colvect(:,:,4)];  tmp_col_sig = tmp_col_sig(:,dots.order_signal(TRIAL, :)); tmp_col_sig = tmp_col_sig(:,~~signaldots_elim);
                
                % Add signal dots below flicker dots
                draw.dotsxy = [tmp_xy_sig tmp_xy];
                draw.colvect = [tmp_col_sig tmp_col];
            case 2
                % randomise order of flickering dots
                tmp_xy = DOT_XY(:,:,TRIAL_TABLE.Col__Attd_UnAttd(TRIAL,1)) ;
                tmp_col = colvect(:,:,TRIAL_TABLE.Col__Attd_UnAttd(TRIAL,1)) ;  
                
                % randomise order of signal dots
                tmp_xy_sig = DOT_XY(:,:,TRIAL_TABLE.Col__Attd_UnAttd(TRIAL,1)+2) ;
                tmp_col_sig = colvect(:,:,TRIAL_TABLE.Col__Attd_UnAttd(TRIAL,1)+2);  
                
                % Add signal dots below flicker dots
                draw.dotsxy = [tmp_xy_sig tmp_xy];
                draw.colvect = [tmp_col_sig tmp_col];
        end
        
%                 draw.dotsxy = [tmp_xy_sig ];
%                 draw.colvect = [tmp_col_sig ];
        
        % draw!
        Screen('DrawDots', windowPtr, draw.dotsxy, dots.width, draw.colvect, mon.centre,1);  % change 1 to 0 to draw square dots
        
        % fixation cross
        %         Screen('FillRect', windowPtr, [colour.red colour.red], dots.fixation );

        Screen('DrawDots', windowPtr, [0;0], dots.width, colour.(str.cue{TRIAL_TABLE.Col__Attd_UnAttd(TRIAL, 1)}), mon.centre,0);  % change 1 to 0 to draw square dots
        
        % Response
        if (FRAME - responseframe) <= f.feedbacklatency
            
            feedbackpos = [
                20 0;
                0 20;
                -20 0;
                0 -20 ];
            Screen('DrawDots', windowPtr, feedbackpos(responsetmp, :)', dots.width*0.75, colour.red, mon.centre,1);  % change 1 to 0 to draw square dots
        end
        
        
        % LINE
        
        if options.arrow && ~isnan(coherentMotionAngle(FRAME,TRIAL, DATA.Col__Attd_UnAttd(TRIAL,1) ))
            arrow1 = [cosd(coherentMotionAngle(FRAME,TRIAL, DATA.Col__Attd_UnAttd(TRIAL,1) )) sind(coherentMotionAngle(FRAME,TRIAL, DATA.Col__Attd_UnAttd(TRIAL,1) ))].*200;
            linepoints = [ mon.centre(1);  mon.centre(2); mon.centre(1) + arrow1(1); mon.centre(2) + arrow1(2)];
            Screen('DrawLine',windowPtr,[0 0 1],linepoints(1),linepoints(2), linepoints(3), linepoints(4), 3)
        end
%         if options.arrow && ~isnan(coherentMotionAngle(FRAME,TRIAL, 2 ))
%             arrow2 = [cosd(coherentMotionAngle(FRAME,TRIAL, 2)) sind(coherentMotionAngle(FRAME,TRIAL, 2))].*200;
%             linepoints = [ mon.centre(1);  mon.centre(2); mon.centre(1) + arrow2(1); mon.centre(2) + arrow2(2)];
%             Screen('DrawLine',windowPtr,[255 255 255],linepoints(1),linepoints(2), linepoints(3), linepoints(4), 3)
%         end
        
        
        % Tell PTB that no further drawing commands will follow before Screen('Flip')
        Screen('DrawingFinished', windowPtr);
        
        
        %% plot some weird flickering things
        %
        %         scaler = linspace(1,0,900);
        %
        %         sig1 = FLICKER(:,:,HH,1);
        %         sig1a = -((-sig1 + baseline).*scaler') + baseline;
        %
        %         sig2 = FLICKER(:,:,HH,2);
        %         sig2a = ((sig2-baseline).*scaler')+baseline;
        %
        %         figure;
        %         hold on;
        %         plot(sig1, 'r')
        %         plot(sig1a, 'k')
        %         plot(sig2, 'g');
        %         plot(sig2a, 'b');
        
        % plot(coherentMotion(:,110,1))
        
        %% flip!
        flipper
        
    end

    %% Present block information
    
    BlockScreen
    
    %% Break?
    if breaker; break; end
end

%% Clean up

sca
figure;
plot(diff(vbl))

%% 
tmp = diff(vbl);
tmp = tmp(:);
tmp(isnan(tmp)) = [];


