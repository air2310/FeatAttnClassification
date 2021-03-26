%% directories

addpath('commonBCIfunctions\')

direct.gNEEDaccessMATLABAPI = 'C:\Program Files\gtec\gNEEDaccessMATLABAPI\';
direct.gNEEDaccess = 'C:\Program Files\gtec\gNEEDaccess\';

direct.main = [ cd '\' ];

% direct.feedback = [ direct.main 'feedback\' ];
% direct.realtime = [ direct.main 'realtime\' ];
% direct.stim = [ direct.feedback 'stim\' ];

direct.dataRoot = [ direct.main 'Data\' ];
direct.resultsRoot = [ direct.main 'Results\' ];

% direct.analysis = [ direct.main 'analysis\' ];
% direct.gtecTopo = [ direct.main 'commonBCIfunctions\gtecTopo\' ];
direct.bufferDRP = [ direct.main 'commonBCIfunctions\bufferDRP\' ];

direct.toolbox = 'E:\toolboxes\';

% direct.cogent = [ direct.toolbox 'Cogent2000v1.33\Toolbox\' ];
direct.io64 =   [ direct.toolbox  'io64\' ];
direct.realtime_hack = [ direct.toolbox 'realtime_hack_07-12-2016\' ];
% direct.morlet_transform_hack = [ direct.toolbox 'morlet_transform_hack\' ];
% direct.controllers = [ direct.toolbox '\MATLAB_Joystick_control-master\' ];

direct.hat = [ direct.toolbox 'hat\' ];

% ----- access functions and scripts

addpath( genpath( direct.gNEEDaccessMATLABAPI ) )
addpath( genpath( direct.gNEEDaccess ) )

% addpath( direct.feedback )
% addpath( direct.realtime )
% addpath( direct.stim )

% addpath( direct.analysis )
% addpath( direct.gtecTopo )
addpath( direct.bufferDRP )

% addpath( direct.cogent ); cgshut;
addpath( direct.io64 )
addpath( direct.realtime_hack ) % allows access to buffer.mexw64 & biosemi2ft
% addpath( direct.morlet_transform_hack )
% addpath( direct.controllers )

addpath( direct.hat )  

