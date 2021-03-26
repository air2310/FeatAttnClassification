%% Setup things
clc
clearvars -except SUB
close all

rng('shuffle')
seed = rng;

%% Observer settings

observer.date = date;
observer.start_clock = clock;

observer.fname = [ observer.date ' ' num2str(observer.start_clock(4)) '-' num2str(observer.start_clock(5)) '-' num2str(observer.start_clock(6)) ];
observer.fname = [ 'S' num2str(SUB) '.' observer.fname ];
observer.fname = strrep( observer.fname, '-', '.' );
observer.fname = strrep( observer.fname, ' ', '.' );

%% Regular directories

str.sub = ['S' num2str(SUB)];
PCuse = 4; % 1 - LAB,  2 - LAPTOP, 3 - BioSemi, 4 - LABNEW

% - Main
switch PCuse
    case 1        
        direct.main = 'E:\';
    case 2     
        direct.main = 'C:\Users\uqarento\Documents\';
    case 3        
        direct.main = 'C:\Users\cogneuro_admin\Documents\Angie\';
    case 4
        direct.main = 'C:\EXPERIMENTS\';
        direct.program = 'C:\Users\labpc\Documents\Angie\RTAttnselectMethods\program\';
end

direct.toolbox = [direct.main 'toolboxes\'];

% - Networking
% direct.ip = [direct.toolbox 'GetIP\']; addpath(direct.ip)
direct.io64 = [direct.toolbox 'io64\']; addpath(direct.io64);
addpath(direct.io64)
% - Data loading directories
% direct.threshold = ['..\minimumMotionStaircase2\data\' str.sub '\'];
% direct.counterbalanceData = [pwd '\Data\'];
% direct.Stim = [pwd '\Stim\'];

% - Saving
direct.data = [direct.program 'data\' str.sub '\']; 
if ~exist(direct.data); mkdir(direct.data);end

direct.results_main = [direct.program  'results\'];
direct.results = [direct.results_main str.sub '\']; 
if ~exist(direct.results); mkdir(direct.results);end

% HAT

direct.hat = [ direct.toolbox 'hat\' ];
addpath( direct.hat )  

% -  Buffer directories
direct.commonBCIfunctions = '\commonBCIfunctions\';
direct.buffer = [pwd direct.commonBCIfunctions 'bufferDRP\'];
direct.realtime_hack = [ direct.toolbox  'realtime_hack_07-12-2016\' ];

addpath( direct.buffer )
addpath(direct.realtime_hack)

% g.tec!

direct.gNEEDaccessMATLABAPI = 'C:\Program Files\gtec\gNEEDaccessMATLABAPI\';
direct.gNEEDaccess = 'C:\Program Files\gtec\gNEEDaccess\';

addpath( genpath( direct.gNEEDaccessMATLABAPI ) )
addpath( genpath( direct.gNEEDaccess ) )
