%% observer

observer.date = date;
observer.start_clock = clock;

observer.fname.session = [ observer.date ' ' num2str(observer.start_clock(4)) '-' num2str(observer.start_clock(5)) '-' num2str(observer.start_clock(6)) ];
observer.fname.session = [ 'S' num2str(observer.number) '.' observer.fname.session ];
observer.fname.session = strrep( observer.fname.session, '-', '.' );
observer.fname.session = strrep( observer.fname.session, ' ', '.' );

% direct.data = [ direct.dataRoot 'S' num2str(observer.number) '\']; mkdir(direct.data);
% direct.results = [ direct.resultsRoot 'S' num2str(observer.number) '\']; mkdir(direct.data);
% 
% mkdir( direct.data )
% mkdir( direct.results )

disp( direct.data )
disp( observer.fname.session )