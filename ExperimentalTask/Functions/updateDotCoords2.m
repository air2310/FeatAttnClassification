function DOT_XY = updateDotCoords2(DOT_XY, dots, DirectFields, directions, TRIAL, FIELD)

dots = genDotHeading(dots,DirectFields, directions, TRIAL, FIELD);
DOT_XY = DOT_XY + dots.dxdy(:,:,FIELD)';%+(rand(2,dots.n)-0.5)*0.5;						% move dots

IDX = find( sqrt( DOT_XY(1,:).^2 + DOT_XY(2,:).^2 ) >= dots.max_d );
if ~isempty(IDX)
    ang = atan2d( DOT_XY(1,IDX), DOT_XY(2,IDX) ) + 180;
    DOT_XY(2,IDX) = ( dots.max_d ).*cosd(ang);
    DOT_XY(1,IDX) = ( dots.max_d ).*sind(ang);
end
end

function  [dots] = genDotHeading(dots,DirectFields, directions, TRIAL, FIELD)

theta_A = directions(DirectFields(TRIAL,FIELD)); % Get Angle
theta_B = dots.SD.*dots.AngDist(:,FIELD, TRIAL) + theta_A; % Create distribution of angles
theta_C = (theta_B./360).*(2*pi); % Convert to radians

dots.dxdy(:,1,FIELD) = dots.speed_frame*cos(theta_C);
dots.dxdy(:,2,FIELD) = dots.speed_frame*sin(theta_C);

end

%% model dots
function modelDots()

diruse = [direct.results 'modelDots\']; mkdir(diruse);
for ii = 1:2:71
    
    model.SD = ii;
    
    model.AngDist = randn(dots.n,1);
    model.theta = 180;
    
    model.theta_B = model.SD.*model.AngDist + model.theta;
    model.theta_C = wrapTo360(model.theta_B);
    
    % figure;
    % plot(sort(model.theta_C))
    
    h = figure;
    hist(model.theta_C, 20)
    xlim([0 360])
    line([1 1].*model.theta, get(gca, 'ylim'), 'color', 'r')
    title(model.SD)
    
    saveas(h, [diruse num2str(model.SD) '.png'])
end
end