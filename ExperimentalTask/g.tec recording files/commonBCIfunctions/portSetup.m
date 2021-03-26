
%% parallel port

addpath('D:\toolboxes\io64')

ioObj = io64;
status = io64(ioObj);

options.port = { 'D010' 'D030' };
%  options.port = { '21' '21' };
n.ports = length(options.port);
address = NaN(n.ports,1);
 
for AA = 1:n.ports
    address(AA) = hex2dec( options.port{AA} );
    io64(ioObj, address(AA), 0);
end

% trig.trial = 1:8; % attend left, attend right, free view

% trig.restTrial = 251;

% trig.initAnalysis = 252;
% trig.stopAnalysis = 253;

trig.stopRecording = 254;
trig.startRecording = 255;

trig.staircase = 55;
% n.trigger_frames = 4;


%% fieldtrip buffer

cfg.host = 'localhost';
cfg.stream = 1982;
cfg.feedback = 9999;