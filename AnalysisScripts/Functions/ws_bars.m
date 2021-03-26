function [ebars] = ws_bars(x)

% within subjects error bars
% assumes subject as row and condition as column

[rows, cols] = size(x);

sub_mean = nanmean(x, 2); %disp( sub_mean );
grand_mean = nanmean( sub_mean, 1 );

x = x - ( repmat( sub_mean, 1, cols ) - grand_mean );

ebars = nanstd( x ) / sqrt( rows ); %nanstd


