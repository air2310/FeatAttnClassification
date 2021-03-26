function dots_xy = SetupDotCoords(dots)

dots.radi = dots.max_d * sqrt(rand(dots.n,1));	% radi of dots
% dots.radi(dots.radi<dots.min_d) = dots.min_d;

theta = 2*pi*rand(dots.n,1);             % theta polar coordinate
cossin = [cos(theta), sin(theta)];
dots_xy = [dots.radi dots.radi] .* cossin;     % dot positions in Cartesian coordinates (pixels from center)

% dots.speed_frame = dots.speed / mon.ref;  % dot speed (pixels/frame)
% dots.dxdy = (rand(dots.n,2)*2*dots.speed_frame)-dots.speed_frame;

% dots.dxdy = (dots.DIR*ones(dots.n,2)*2*dots.speed_frame)-dots.speed_frame;
