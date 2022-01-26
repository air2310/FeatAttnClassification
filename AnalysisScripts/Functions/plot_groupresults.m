function plot_groupresults(ACCMEAN_ALL, sets, runopts, decodestring)
% plot_groupresults
% classifier listed
%   Inputs:
%       ACCMEAN_ALL - structure containing accuracy across all subjects and
%       training conditions for input classifier.
%       sets - structure of metadata and settings
%       runopts - run options for this decoding run
%       decodestring - which decoder to run. Options: MLP, LDA, KNN,

%   Outputs:
%       ACCURACY - structure containing accuracy across all subjects and
%       training conditions for input classifier.

%% Plot!

h = figure;
for ii_hzstate = 1:sets.n.hzstates
    subplot(2,2,ii_hzstate)
    
    % Get data
    datplot = ACCMEAN_ALL(:,:,ii_hzstate,:);
    M = mean(datplot, 4)';
    E = NaN(sets.n.chunksizes, sets.n.cols);
    for ii_col = 1:sets.n.cols
        E(:, ii_col) = ws_bars(squeeze(datplot(ii_col, :, :, :))');
    end
    x = [sets.timing.secs.chunksizes; sets.timing.secs.chunksizes]';
    
    % Plot it
    hold on
    
    errorbar(x(:,1), M(:,1), E(:,1), 'k-')
    errorbar(x(:,2), M(:,2), E(:,2), 'k:')
    line([0 4], [50 50], 'color','r')
    ylim([48 100])
    
%     	set(gca, 'xtick', sets.timing.secs.chunksizes, 'xticklabel', sets.timing.secs.chunksizes)
    xlabel('Sliding window size')
    ylabel('Accuracy (%)')
    legend(sets.str.colcond, 'location', 'NorthWest')
    title(sets.str.HzState{ii_hzstate})
    
end

trainstring = ['Train' sets.str.testtrainopts{runopts.traindat} 'Test' sets.str.testtrainopts{runopts.testdat} sets.str.excludemotepochs{runopts.excludemotepochs}];

tit = ['Group ' decodestring ' Accuracy ' trainstring];
suptitle(tit);
saveas(h, [sets.direct.results_group  tit '.png'])

end