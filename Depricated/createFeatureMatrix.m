function [randomized_feature_matrix, labels_in] = createFeatureMatrix(FootSt, labels_train, k, prediction_signals, two_feet, foot_names)
    strideList = FootSt.(foot_names{k});
    n2= setdiff(foot_names,foot_names{k});
    strideListOther = FootSt.(n2{:});
    if two_feet
        lenStrideList = min(length(strideList), length(strideListOther));
    else
        lenStrideList = length(strideList);
    end
    lab_foot = labels_train([strideList.globalInitStanceSample]);
%     disp('Extracting predictor features...');
    nOutputSignals = numel(prediction_signals)*(two_feet+1);
    
    means = zeros(lenStrideList, numel(prediction_signals)*(two_feet+1));
    maxs = means;
    mins = means;
    ranges = means;
    
    jrange = 1:(two_feet+1):nOutputSignals;
    start = 1;
    
%     handles.axes1 = axes;
%     set(handles.axes1, 'NextPlot', 'add');
    for i=1:lenStrideList
        
%         window_start = strideList(i).globalFootStaticSample - strideList(i).globalInitStanceSample + 1;
%         window_end = strideList(i).globalInitSwingSample - strideList(i).globalInitStanceSample + POST_SWING_CUTOFF_SAMPLES;
        iter = 1;
        lsend = start + length(strideList(i).(prediction_signals{1}));
%         lsend = start + 10*length(strideList(i).(prediction_signals{1}));
%         N = length(strideList(i).(prediction_signals{1}));
%         timepts = linspace(start,lsend,N-1);
%         timepts = ttot(start:lsend-1);
%         labs = labels_train(start:lsend-1);
        start = lsend(end);
        
        
        for jj=1:length(jrange)
            j = jrange(jj);
            
            cycle_time = length(strideList(i).(prediction_signals{iter}));
            % start after planatar flexion
            window_start = 1; 
%             window_end = .6*cycle_time - POST_SWING_CUTOFF_SAMPLES;
            window_end = cycle_time;
%             window_start = strideList(i).globalFootStaticSample - strideList(i).globalInitStanceSample + 1;
%             if i < lenStrideList
%                 window_end = strideList(i+1).globalInitSwingSample - strideList(i).globalInitStanceSample + POST_SWING_CUTOFF_SAMPLES;
%             else
%                 window_end = size(data_train, 1);
%             end
            means(i,j) = mean(strideList(i).(prediction_signals{iter})(window_start:window_end));
            maxs(i,j) = max(strideList(i).(prediction_signals{iter})(window_start:window_end));
            mins(i,j) = min(strideList(i).(prediction_signals{iter})(window_start:window_end));            
            ranges(i,j) = range(strideList(i).(prediction_signals{iter})(window_start:window_end));
            
%             if plot_pts
%                 figure(iter);
%                 hold on;
%                 plot(timepts,strideList(i).(prediction_signals{iter}),gp_mat{mode(labs(labs>0))})
%                 hold on;
%                 ax = gca;
%                 line([ttot(strideList(i).globalInitStanceSample) ...
%                     ttot(strideList(i).globalInitStanceSample)],get(ax,'YLim'),...
%                     'Color',[1 0 0]);
% %                 line([ttot(strideList(i).globalFootStaticSample) ...
% %                     ttot(strideList(i).globalFootStaticSample)],get(ax,'YLim'),...
% %                     'Color',[0 1 0]);
% %                 line([ttot(strideList(i).globalInitSwingSample) ...
% %                     ttot(strideList(i).globalInitSwingSample)],get(ax,'YLim'),...
% %                     'Color',[0 0 1]);
%                 
%                 if i == lenStrideList
%                     legend('Strides','InitStance','Static','InitSwing');
%                 end
%             end
%             handles.handle_plotCD(i) = plot(timepts,...
%                 strideList(i).(prediction_signals{iter}),'parent',handles.axes1);
%             if i == 1
%                 title(prediction_signals{iter})
%             end
%             hold off;
            if two_feet
                
                cycle_time = length(strideListOther(i).(prediction_signals{iter}));
                % start after planatar flexion
                window_start = 1; 
%                 window_end = .6*cycle_time - POST_SWING_CUTOFF_SAMPLES;
                window_end = cycle_time;
                
%                 window_start = strideListOther(i).globalFootStaticSample - strideListOther(i).globalInitStanceSample + 1;
%                 window_end = strideListOther(i).globalInitSwingSample - strideListOther(i).globalInitStanceSample + POST_SWING_CUTOFF_SAMPLES;

                means(i,j+1) = mean(strideListOther(i).(prediction_signals{iter})(window_start:window_end));
                maxs(i,j+1) = max(strideListOther(i).(prediction_signals{iter})(window_start:window_end));
                mins(i,j+1) = min(strideListOther(i).(prediction_signals{iter})(window_start:window_end));
                ranges(i,j+1) = range(strideListOther(i).(prediction_signals{iter})(window_start:window_end));
            end
            iter = iter+1;
        end
    end
    feature_matrix = [maxs, mins, ranges];
    
    rperm = randperm(lenStrideList);
    randomized_feature_matrix = feature_matrix(rperm,:);

    labels_in = lab_foot(rperm);

end