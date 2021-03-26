labels = { 'P010' 'P10' 'P08' 'O10' 'O2' 'PO4' 'Iz' 'Oz' 'POz' 'Pz' 'O9' 'O1' 'PO3' 'PO9' 'PO7' 'P9' };

IM2 = imread( 'C:\Users\solc-lab02\Desktop\gtecTopos\gtecHead.bmp' );
IM = imread( 'C:\Users\solc-lab02\Desktop\gtecTopos\electrodePositions.bmp' );
IM = fliplr(IM);

x = [];
y = [];

for CC = 1:16
    
    mask = IM(:,:,1)==CC;
    
    IND = find( mask );
    [I,J] = ind2sub( size( IM(:,:,1) ),IND);
    y(CC) = size(IM,1) - mean(I);
    x(CC) =  size(IM,2) - mean(J);
    
end

% figure; hold on
% image( [144 size(IM,2)-131]-10, [129 size(IM,1)-22], flipud(IM2)); hold on
% axis equal
% xlim( [1 size(IM,2)] )
% ylim( [1 size(IM,1)] )
% scatter(x,y)
% 
% for CC = 1:length(labels)
%     text( x(CC), y(CC), labels{CC}, 'horizontalalignment', 'center', 'verticalalignment', 'middle' )
% end

gtecChanlocs.IM = IM;
gtecChanlocs.IM2 = IM2;
gtecChanlocs.x = x;
gtecChanlocs.y = y;
gtecChanlocs.labels = labels;

save( 'gtecChanlocs.mat', 'gtecChanlocs' )