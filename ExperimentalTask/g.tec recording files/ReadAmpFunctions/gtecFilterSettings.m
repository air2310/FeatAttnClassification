
load( 'filters.mat', 'BandpassFilters', 'NotchFilters', 'F', 'supported_fs' )

%              FilterIndex: 1
%             SamplingRate: 2
%                    Order: 3
%     LowerCutoffFrequency: 4
%     UpperCutoffFrequency: 5

if options.NumberOfScans ~= 0
    NumberOfScans = options.NumberOfScans;
else
    NumberOfScans = supported_fs( 2, supported_fs(1,:) == fs ); % default value
end

% ( 1000 ./ supported_fs(1,:) ) .* supported_fs(2,:)

switch fs
    case 256
        BandpassFilterIndex = 47;   % 47         256           8           1         100 
        NotchFilterIndex    = 2;    % 2          256           4          48          52
    case 512
        BandpassFilterIndex = 72; % 72     512       8       1     100
        NotchFilterIndex = 4; % 4     512       4      48      52
    case 1200
        BandpassFilterIndex = 132; % 132    1200       8       1     100
        NotchFilterIndex = 8; % 8    1200       4      48      52
    case 38400
        BandpassFilterIndex = 363;  % 363       38400       4       1     100
        NotchFilterIndex = 18;      % 18        38400       4      48      52
    otherwise
        BandpassFilterIndex = -1; % (-1 = no filter)
        NotchFilterIndex = -1; % (-1 = no filter)
end

if ~options.filter
    BandpassFilterIndex = -1; % (-1 = no filter)
    NotchFilterIndex = -1; % (-1 = no filter)
end

% NotchFilterIndex = 2;