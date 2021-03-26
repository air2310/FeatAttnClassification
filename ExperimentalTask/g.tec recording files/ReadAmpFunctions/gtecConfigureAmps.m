
gds_interface = gtecDeviceInterface;

gds_interface.IPAddressHost = '127.0.0.1';
gds_interface.IPAddressLocal = '127.0.0.1';
gds_interface.HostPort = 50223;
gds_interface.LocalPort = 50224;

connected_devices = gds_interface.GetConnectedDevices();
N.connected_devices = length( connected_devices );

gusbamp_configs( 1, 1:N.connected_devices ) = gUSBampDeviceConfiguration();

%% set device order

N.amplifiers = 4;
N.gUSBampChannels = 16;

ampChanIdx(1,:) = 1:16;
ampChanIdx(2,:) = 17:32;
ampChanIdx(3,:) = 33:48;
ampChanIdx(4,:) = 49:64;

channelNames = cell(1,N.amplifiers);

for i = 1:N.amplifiers
    for j = 1:N.gUSBampChannels
        channelNames{i}{j} = num2str( ampChanIdx(i,j) );
    end
end

switch N.connected_devices
    case 4
        gusbamp_configs(1,1).Name = 'UB-2016.08.23'; % master ( master must be #1 or crash! ) - SYNC OUT
        gusbamp_configs(1,2).Name = 'UB-2016.08.22'; % slave - SYNC IN
        gusbamp_configs(1,3).Name = 'UB-2016.08.21'; % slave - SYNC IN
        gusbamp_configs(1,4).Name = 'UB-2016.08.20'; % slave - SYNC IN
    case 3
        gusbamp_configs(1,1).Name = 'UB-2016.08.23'; % master ( master must be #1 or crash! ) - SYNC OUT
        gusbamp_configs(1,2).Name = 'UB-2016.08.22'; % slave - SYNC IN
        gusbamp_configs(1,3).Name = 'UB-2016.08.21'; % slave - SYNC IN
    case 2
        gusbamp_configs(1,1).Name = 'UB-2016.08.20'; % master ( master must be #1 or crash! ) - SYNC OUT
        gusbamp_configs(1,2).Name = 'UB-2016.08.21'; % slave - SYNC IN
    case 1
        gusbamp_configs(1,1).Name = 'UB-2016.08.20'; % master ( master must be #1 or crash! ) - SYNC OUT
end

gds_interface.DeviceConfigurations = gusbamp_configs;



%% channels to acquire

N.channels2acquire = length( idx.channels2acquire ) + options.TriggerEnabled; % + 2 x printer port channels & read counter for saving and analysis

channels2acquire = ismember( 1:N.gUSBampChannels, idx.channels2acquire );

%% configure amplifers & channels

available_channels = cell( 1, N.connected_devices );

for i = 1 : N.connected_devices
    
    disp( [ 'configuring... ' connected_devices(1,i).Name ] )
    
    available_channels{i} = gds_interface.GetAvailableChannels( gusbamp_configs(1,i).Name );
    
    % ----- SamplingRate & NumberOfScans
    
    gusbamp_configs(1,i).SamplingRate = fs;
    gusbamp_configs(1,i).NumberOfScans = NumberOfScans;
    
    % ----- CommonGround & CommonReference
    
    gusbamp_configs(1,i).CommonGround = options.CommonGround;
    gusbamp_configs(1,i).CommonReference = options.CommonReference;
    
    % ----- InternalSignalGenerator
    
    gusbamp_configs(1,i).InternalSignalGenerator = gusbamp_siggen;
    
    % ----- ShortCutEnabled, CounterEnabled & TriggerEnabled
    
    gusbamp_configs(1,i).ShortCutEnabled = options.ShortCutEnabled;
    gusbamp_configs(1,i).CounterEnabled = options.CounterEnabled;
    gusbamp_configs(1,i).TriggerEnabled = options.TriggerEnabled;
    
    % ----- individual channel settings
    
    for j = 1 : size( gusbamp_configs(1,i).Channels, 2)
        if ( available_channels{i}(1,j) )
            
            % ----- recording
            gusbamp_configs(1,i).Channels(1,j).Available = true; % don't know what this does
            gusbamp_configs(1,i).Channels(1,j).Acquire = channels2acquire(j); % if false, channel not acquired in the read
            
            % ----- filters
            gusbamp_configs(1,i).Channels(1,j).BandpassFilterIndex = BandpassFilterIndex;
            gusbamp_configs(1,i).Channels(1,j).NotchFilterIndex = NotchFilterIndex;
            
            % ----- bipolar channels
            gusbamp_configs(1,i).Channels(1,j).BipolarChannel = 0; % do not use a bipolar channels
            
        end
    end
    
end

gds_interface.DeviceConfigurations = gusbamp_configs;
gds_interface.SetConfiguration();

