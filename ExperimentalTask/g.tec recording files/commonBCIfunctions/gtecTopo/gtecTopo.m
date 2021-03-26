function gtecTopo( head, gtecChanlocs, electrodes, idxBest, cm, limit, ds, zoom )

IM = gtecChanlocs.IM;
IM2 = gtecChanlocs.IM2;
x = gtecChanlocs.x;
y = gtecChanlocs.y;
labels = gtecChanlocs.labels;

hold on

% ----- head

image( [144 size(IM,2)-131]-10, [129 size(IM,1)-22], imcomplement( flipud(IM2)) ); hold on

xlim( [1 size(IM2,2)] )
ylim( [1 size(IM2,1)] )

% ----- 2D interpolation (heart of topo)

[xq,yq] = meshgrid( min(x):max(x), min(y):max(y));
vq = griddata(x,y,head,xq,yq,'cubic');
surf( xq, yq, vq, 'linestyle', 'none', 'facecolor', 'interp' )

% ---- electrodes

switch electrodes
    case 'on'
        scatter3(x, y, repmat(max(head)*1.1,length(x),1), ds, 'g', 'f')
        
        if ~isempty( idxBest )
            scatter3( x(idxBest), y(idxBest), repmat(max(head)*1.1,length(idxBest),1), ds, 'w', 'f')
        end
        
    case 'labels'
        for CC = 1:length(labels)
            text( x(CC), y(CC), max(head)*1.1, labels{CC}, 'horizontalalignment', 'center', 'verticalalignment', 'middle', 'fontsize', 12, 'color', 'g' )
        end
end

colormap(cm)

hc = colorbar;
set(get(hc,'title'),'string','\muV');
set(hc,'xtick', [ min(limit) max(limit) ] )

axis off equal

if ~isempty(zoom)
    set(gca,'xlim',zoom.x)
    set(gca,'ylim',zoom.y)
end
    
