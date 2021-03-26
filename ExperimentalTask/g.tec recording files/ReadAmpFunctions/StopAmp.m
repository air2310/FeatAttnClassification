if exist( 'gds_interface', 'var' ) % turn off if running
    try
        disp('stopping aquisition...')
        tic
        gds_interface.StopDataAcquisition();
        toc
        pause(.5);
    catch
    end
    delete( gds_interface ); clear gds_interface;
end

% addpath('commonBCIfunctions\')