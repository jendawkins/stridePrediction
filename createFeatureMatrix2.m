function [feature_matrix] = createFeatureMatrix2(strideList, strideListOther, prediction_signals, two_feet)

    POST_SWING_CUTOFF_SAMPLES = 15;
    if two_feet
        lenStrideList = min(length(strideList), length(strideListOther));
    else
        lenStrideList = length(strideList);
    end
%     lab_foot = labels_train([strideList.globalInitStanceSample]);
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
    lone_sig = [];
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
        
        lone_iter = 1;
        delj = [];
        for jj=1:length(jrange)
            j = jrange(jj);
            
            cycle_time = length(strideList(i).(prediction_signals{iter}));
            
            
            if cycle_time == 1
                lone_sig(i,lone_iter) = strideList(i).(prediction_signals{iter});
                lone_iter = lone_iter+1;
                delj = [delj, j];
            else
            % start after planatar flexion
%             window_start = 1; 
%             window_end = round(.6*cycle_time - POST_SWING_CUTOFF_SAMPLES);
            window_start = round(.6*cycle_time);
            window_end = window_start + 10;
%             window_end = cycle_time;
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
            end
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
                
                % Start and second foot SWING
                window_start = 1; 
%                 window_end = round(.6*cycle_time - POST_SWING_CUTOFF_SAMPLES);
%                   window_start = round(.6*cycle_time);
                   
                  window_end = window_start + 10;
%                 window_end = cycle_time;
                
%                 window_start = strideListOther(i).globalFootStaticSample - strideListOther(i).globalInitStanceSample + 1;
%                 window_end = strideListOther(i).globalInitSwingSample - strideListOther(i).globalInitStanceSample + POST_SWING_CUTOFF_SAMPLES;
                if cycle_time == 1
                    lone_sig(i,lone_iter) = strideListOther(i).(prediction_signals{iter});
                    lone_iter = lone_iter+1;
                    delj = [delj, j];
                else
                means(i,j+1) = mean(strideListOther(i).(prediction_signals{iter})(window_start:window_end));
                maxs(i,j+1) = max(strideListOther(i).(prediction_signals{iter})(window_start:window_end));
                mins(i,j+1) = min(strideListOther(i).(prediction_signals{iter})(window_start:window_end));
                ranges(i,j+1) = range(strideListOther(i).(prediction_signals{iter})(window_start:window_end));
            
                end
            end
            iter = iter+1;
        end
    end
    if ~isempty(lone_sig)

    end
    maxs(:,delj)=[]; mins(:,delj) = []; ranges(:,delj) = [];
    feature_matrix = [maxs, mins, ranges, lone_sig];
%     feature_matrix = [maxs(:,1:end-size(lone_sig,2)), mins(:,1:end-size(lone_sig,2)), ...
%         ranges(:,1:end-size(lone_sig,2)), lone_sig];
%     if size(feature_matrix,2)>30
%         x = 'debug';
%     end
    
%     rperm = randperm(lenStrideList);
%     randomized_feature_matrix = feature_matrix(rperm,:);
% 
%     labels_in = lab_foot(rperm);

end