N.amplifiers = 4;
N.gUSBampChannels = 16;

gusbamp_configs(1,1).Name = 'UB-2016.08.23'; % master ( master must be #1 or crash! ) - SYNC OUT
gusbamp_configs(1,2).Name = 'UB-2016.08.22'; % slave - SYNC IN
gusbamp_configs(1,3).Name = 'UB-2016.08.21'; % slave - SYNC IN
gusbamp_configs(1,4).Name = 'UB-2016.08.20'; % slave - SYNC IN 

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