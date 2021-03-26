if ismember(TRIAL,blockendtrials)
    trialstage = 0;
        % Calculate performance for last block
        if BLOCK >1
            trialspastblock = TRIAL_TABLE.TRIAL(TRIAL_TABLE.BLOCK==(BLOCK-1));
            
            ACC(:,trialspastblock) = zeros(n.dirChangesTrial,n.trialsBlock);
            for TT = 1:n.trialsBlock
                
                % get responses
                dat = RESPONSE(:, trialspastblock(TT));
                tmp_responseframes = find([0; diff(dat)]>1);
                tmp_responses = dat(tmp_responseframes);
                
                % get correct responses
                possibleresponseperiods = NaN(2,n.dirChangesTrial);   
                frames = TRIAL_TABLE.moveframe__Attd_UnAttd{trialspastblock(TT),1};
                
                for RR = 1:n.dirChangesTrial
                    % get  possible response periods
                    possibleresponseperiods(1,RR) = frames(RR);
                    if RR < n.dirChangesTrial
                        possibleresponseperiods(2,RR) = frames(RR+1);
                    else
                        possibleresponseperiods(2,RR) = f.trial;
                    end
                    
                    % evaluate if a response happened then, and if it was
                    % correct
                    tmp_idx = find(tmp_responseframes >  possibleresponseperiods(1,RR) & tmp_responseframes <  possibleresponseperiods(2,RR));
                    if any(tmp_idx)
                        reorder = [1 2 3 4 ];
                       
                        if tmp_responses(tmp_idx) == find(ismember(directions(reorder),  TRIAL_TABLE.movedir__Attd_UnAttd{trialspastblock(TT),1}(RR)))
                           
                            ACC(RR,trialspastblock(TT)) = 1;
                            RT(RR,trialspastblock(TT)) = (tmp_responseframes(tmp_idx) -  frames(RR))/mon.ref;
                        else
                            ACC(RR,trialspastblock(TT)) = 2;
                        end
                    end
                end
            end
            
            tmp = ACC(:,trialspastblock);
            acctmp = 100*sum(tmp(:)==1)/length(tmp(:));
            inacctmp = 100*sum(tmp(:)==2)/length(tmp(:));
            misses = 100*sum(tmp(:)==0)/length(tmp(:));
            RTtmp = round(100*nanmean(nanmean(RT(:,trialspastblock))))/100;
        end

        % Display block info
        
        
        
        while true
            
            % present progress
            Screen('TextFont',windowPtr, 'Arial Black');
            Screen('TextSize',windowPtr, 40);
            
            BLOCK = TRIAL_TABLE.BLOCK(TRIAL);
            str.block = ['BLOCK: ' num2str(BLOCK) ' of ' num2str(n.blocks) ];
            DrawFormattedText(windowPtr,str.block, 'center', mon.centre(2) - 250,colour.BLACK);
            
             % present advance Advance
            DrawFormattedText(windowPtr,'Press [ENTER]', 'center', mon.centre(2) - 150 ,colour.BLACK);
            
            
            % Present feedback
           colour.green2 = uint8([10 100 0]);
            
            str.acc = ['Correct Responses: ' num2str(acctmp) '%'];
            str.inacc = ['Incorrect Responses: ' num2str(inacctmp) '%'];
            str.miss = ['Missed Responses: ' num2str(misses) '%'];
            str.RT = ['Reaction Time: ' num2str(RTtmp) 's'];
            
            Screen('TextSize',windowPtr, 30);
            DrawFormattedText(windowPtr,'Performance on previous block:' , 'center', mon.centre(2) + 50,colour.green2);
            
            Screen('TextFont',windowPtr, 'Arial');
            
            DrawFormattedText(windowPtr,str.acc, 'center', mon.centre(2) + 100,colour.green2);
            DrawFormattedText(windowPtr,str.inacc, 'center', mon.centre(2) + 150,colour.green2);
            DrawFormattedText(windowPtr,str.miss, 'center', mon.centre(2) + 200,colour.green2);
            DrawFormattedText(windowPtr,str.RT, 'center', mon.centre(2) +250,colour.green2);
            
           
            % flip
            flipper
            
            % break if enter
            [~, ~, keyCode, ~] = KbCheck();
            if any(find(keyCode)==key.enter)
                break;
            end
        end
        Screen('TextFont',windowPtr, 'Arial Black');
        Screen('TextSize',windowPtr, 60);
    end
       