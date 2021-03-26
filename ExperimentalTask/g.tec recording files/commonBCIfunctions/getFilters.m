if exist( 'gds_interface', 'var' )
    gds_interface.StopDataAcquisition();
    delete(gds_interface);
    clear gds_interface;
end

clear
clc
close all

gds_interface = gtecDeviceInterface;

gds_interface.IPAddressHost = '127.0.0.1';
gds_interface.IPAddressLocal = '127.0.0.1';
gds_interface.HostPort = 50223;
gds_interface.LocalPort = 50224;

connected_devices = gds_interface.GetConnectedDevices();

gusbamp_configs = gUSBampDeviceConfiguration();

gusbamp_configs.Name = connected_devices(1,1).Name;
gds_interface.DeviceConfigurations = gusbamp_configs;

supported_fs = gds_interface.GetSupportedSamplingRates( gusbamp_configs.Name );

BandpassFilters = [];
NotchFilters = [];

for sampling_rate = supported_fs(1,:)
    
    disp( sampling_rate )

    available_filters = gds_interface.GetAvailableFilters(sampling_rate);

    BandpassFilters = [ BandpassFilters ;
    [ available_filters.BandpassFilters.FilterIndex ]' ...
    [ available_filters.BandpassFilters.SamplingRate ]' ...
    [ available_filters.BandpassFilters.Order ]' ...
    [ available_filters.BandpassFilters.LowerCutoffFrequency ]' ...
    [ available_filters.BandpassFilters.UpperCutoffFrequency ]' ];

    NotchFilters = [ NotchFilters ;
    [ available_filters.NotchFilters.FilterIndex ]' ...
    [ available_filters.NotchFilters.SamplingRate ]' ...
    [ available_filters.NotchFilters.Order ]' ...
    [ available_filters.NotchFilters.LowerCutoffFrequency ]' ...
    [ available_filters.NotchFilters.UpperCutoffFrequency ]' ];

end

F.FilterIndex = 1;
F.SamplingRate = 2;
F.Order = 3;
F.LowerCutoffFrequency = 4;
F.UpperCutoffFrequency = 5;

save( 'filters.mat', 'BandpassFilters', 'NotchFilters', 'F', 'supported_fs' )
