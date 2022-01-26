function plot_subresults(ACCMEAN, sets, decodestring, runopts)
% plotresults
%   Inputs:
%       ACCMEAN - mean accuracy results across conditions
%       sets - settings structure
%       decodestring - string describing which decoder this was. 
%       runopts - options for this run. 

%% Plot results
h = figure;
count = 0;
for ii_hzstate = 1:sets.n.hzstates
    count = count + 1;
    subplot(2,2,count)
    
    datplot = ACCMEAN(:,:,ii_hzstate)';
    plot(sets.timing.secs.chunksizes, datplot, '-x')
    ylim([48 100])
    
    % 	set(gca, 'xtick', sets.timing.secs.chunksizes, 'xticklabel', sets.timing.secs.chunksizes)
    xlabel('Sliding window size')
    ylabel('Accuracy (%)')
    legend(sets.str.colcond, 'location', 'NorthWest')
    title(sets.str.HzState{ii_hzstate})
    
end

trainstring = ['Train' sets.str.testtrainopts{runopts.traindat} 'Test' sets.str.testtrainopts{runopts.testdat} sets.str.excludemotepochs{runopts.excludemotepochs}];

tit = [sets.str.sub ' ' decodestring ' Accuracy'];
suptitle(tit);
saveas(h, [sets.direct.resultssub tit trainstring '.png'])

end