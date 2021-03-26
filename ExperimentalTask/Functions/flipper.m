%% Flip
if trialstage == 2
    vbl(FRAME, TRIAL) = Screen('Flip', windowPtr);
else
    Screen('Flip', windowPtr);
end

%% Trigger

i.Hzattd = TRIAL_TABLE.Freq__Attd_UnAttd(TRIAL,1);
i.Colattd = TRIAL_TABLE.Col__Attd_UnAttd(TRIAL,1);
i.attcond = TRIAL_TABLE.ATTENTIONCOND(TRIAL);

if options.trigger
    % ## CUE ##
    if FRAME_cue  == 1
        % Hz Attend, Col Attend, Training Cond
        trigsend = trig.cue(i.Hzattd,i.Colattd,i.attcond);
    end
    
    % ## TRIAL ##
    if FRAME == 1
         %  Hz Attend, Col Attend, Training Cond
         trigsend = trig.trial(i.Hzattd,i.Colattd,i.attcond);
    end
    
    % ## MOTION ##
    Attn_idx = 1;
    % Attd/unatted, Hz move, Col move, training cond, motion direction
    if ismember(FRAME, TRIAL_TABLE.moveframe__Attd_UnAttd{TRIAL,Attn_idx})
        % NB - movedir looks wrong
        movedir_idx = find(ismember(FRAME, TRIAL_TABLE.moveframe__Attd_UnAttd{TRIAL,Attn_idx}));
        i.Colmove = TRIAL_TABLE.Col__Attd_UnAttd(TRIAL, Attn_idx);
        
        trigsend = trig.motion(i.Hzattd,i.Colattd,i.attcond, movedir_idx);
        
    end

    % ## FEEDBACK ##
    if FRAME_feedback == 1
         trigsend =  trig.feedback;
    end
    
    %% TRIGGER
    io64(trig.ioObj, trig.address(2), trigsend);
    
end

%% Capture
%      imageArray(:,:,:,FRAME_ALL)= Screen('GetImage', windowPtr, [mon.centre-350 mon.centre+350]);