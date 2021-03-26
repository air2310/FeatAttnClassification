function sendParallel( FRAME, n, trig2use, ioObj, address )

% ----- parallel trigger

n.trigger_frames = 4;
 
if ismember( FRAME, 1 : n.trigger_frames )
    TRIG = trig2use;
else
    TRIG = 0;
end

io64(ioObj, address, TRIG);

    
