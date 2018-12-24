function [acc, C]=crossval_fun(data_train,labels_train, data_test, labels_test, plot_pts, two_feet, ttot)
POST_SWING_CUTOFF_TIME_S=0.2;
SAMPLE_RATE_HZ = 100;
POST_SWING_CUTOFF_SAMPLES = int32(POST_SWING_CUTOFF_TIME_S * SAMPLE_RATE_HZ);

fin_mat3 = [data_train; data_test];
% ttot = linspace(0, 10*size(fin_mat3,1)-1, size(fin_mat3,1)-1);
z = find(fin_mat3(:,7)==1);
z2 = find(fin_mat3(:,end)==1);

y = z(2:end)-z(1:end-1);
y2 = z2(2:end)-z2(1:end-1);

cycle_time = mean(y(y<prctile(y,75) & y>prctile(y,25)));
[FootSt] = romanFunction(data_train);
inverse_kinematics_outputs = {'r0','r1','r2','r3','gAccY','gAccZ','gVelY','gVelZ','gPosY','gPosZ'};

prediction_signals = ['aAccY','aAccZ','aOmegaX','d1aAccY','d1aAccZ','d1aOmegaX','i1aAccY','i1aAccZ','i1aOmegaX',inverse_kinematics_outputs];
foot_names= {'F1','F2'};
feature_matrix =[];
lab_both = [];

W = [];
%% Training
% lab_foot = labels_train;

% figure(1);
gp_mat = {'m','c','y'};
for k = 1:two_feet+1
    [randomized_feature_matrix, labels_in] = createFeatureMatrix(FootSt, labels_train, k, prediction_signals, two_feet, foot_names);
%     strideList = FootSt.(foot_names{k});
%     n2= setdiff(foot_names,foot_names{k});
%     strideListOther = FootSt.(n2{:});
%     if two_feet
%         lenStrideList = min(length(strideList), length(strideListOther));
%     else
%         lenStrideList = length(strideList);
%     end
%     lab_foot = labels_train([strideList.globalInitStanceSample]);
% %     disp('Extracting predictor features...');
%     nOutputSignals = numel(prediction_signals)*(two_feet+1);
%     
%     means = zeros(lenStrideList, numel(prediction_signals)*(two_feet+1));
%     maxs = means;
%     mins = means;
%     ranges = means;
%     
%     jrange = 1:(two_feet+1):nOutputSignals;
%     start = 1;
%     
% %     handles.axes1 = axes;
% %     set(handles.axes1, 'NextPlot', 'add');
%     for i=1:lenStrideList
%         
% %         window_start = strideList(i).globalFootStaticSample - strideList(i).globalInitStanceSample + 1;
% %         window_end = strideList(i).globalInitSwingSample - strideList(i).globalInitStanceSample + POST_SWING_CUTOFF_SAMPLES;
%         iter = 1;
%         lsend = start + length(strideList(i).(prediction_signals{1}));
% %         lsend = start + 10*length(strideList(i).(prediction_signals{1}));
% %         N = length(strideList(i).(prediction_signals{1}));
% %         timepts = linspace(start,lsend,N-1);
%         timepts = ttot(start:lsend-1);
%         labs = labels_train(start:lsend-1);
%         start = lsend(end);
%         
%         
%         for jj=1:length(jrange)
%             j = jrange(jj);
%             
%             cycle_time = length(strideList(i).(prediction_signals{iter}));
%             % start after planatar flexion
%             window_start = 1; window_end = .6*cycle_time - POST_SWING_CUTOFF_SAMPLES;
%             window_end = cycle_time;
% %             window_start = strideList(i).globalFootStaticSample - strideList(i).globalInitStanceSample + 1;
% %             if i < lenStrideList
% %                 window_end = strideList(i+1).globalInitSwingSample - strideList(i).globalInitStanceSample + POST_SWING_CUTOFF_SAMPLES;
% %             else
% %                 window_end = size(data_train, 1);
% %             end
%             means(i,j) = mean(strideList(i).(prediction_signals{iter})(window_start:window_end));
%             maxs(i,j) = max(strideList(i).(prediction_signals{iter})(window_start:window_end));
%             mins(i,j) = min(strideList(i).(prediction_signals{iter})(window_start:window_end));            
%             ranges(i,j) = range(strideList(i).(prediction_signals{iter})(window_start:window_end));
%             
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
% %             handles.handle_plotCD(i) = plot(timepts,...
% %                 strideList(i).(prediction_signals{iter}),'parent',handles.axes1);
%             if i == 1
%                 title(prediction_signals{iter})
%             end
%             hold off;
%             if two_feet
%                 
%                 cycle_time = length(strideListOther(i).(prediction_signals{iter}));
%                 % start after planatar flexion
%                 window_start = 1; 
% %                 window_end = .6*cycle_time - POST_SWING_CUTOFF_SAMPLES;
%                 window_end = cycle_time;
%                 
% %                 window_start = strideListOther(i).globalFootStaticSample - strideListOther(i).globalInitStanceSample + 1;
% %                 window_end = strideListOther(i).globalInitSwingSample - strideListOther(i).globalInitStanceSample + POST_SWING_CUTOFF_SAMPLES;
% 
%                 means(i,j+1) = mean(strideListOther(i).(prediction_signals{iter})(window_start:window_end));
%                 maxs(i,j+1) = max(strideListOther(i).(prediction_signals{iter})(window_start:window_end));
%                 mins(i,j+1) = min(strideListOther(i).(prediction_signals{iter})(window_start:window_end));
%                 ranges(i,j+1) = range(strideListOther(i).(prediction_signals{iter})(window_start:window_end));
%             end
%             iter = iter+1;
%         end
%     end
%     feature_matrix = [maxs, mins, ranges];
% 
%     rperm = randperm(lenStrideList);
%     randomized_feature_matrix = feature_matrix(rperm,:);
% 
%     labels_in = lab_foot(rperm);
    
    Mdl.(['Foot' num2str(k)]) = fitcdiscr(randomized_feature_matrix,labels_in);
%     W(:,:,k) = LDA(randomized_feature_matrix, labels_in);
end
% nOutputFeatures = nOutputSignals * 3;
% [W] = batchProcessor3(data_in, labels, new_time);
% start_ind = round(max([find(data_in(:,7)==1,1);find(data_in(:,end)==1,1)]))+1;
% data_past = data_in(1:start_ind-1,:);
k = 1;
FootSt2 = romanFunction(data_test);

[randomized_feature_matrix_test, labels_in_test] = createFeatureMatrix(FootSt2, labels_test, k, prediction_signals, two_feet, foot_names);

guess_train = predict(Mdl.(['Foot' num2str(k)]), randomized_feature_matrix);
guesses = predict(Mdl.(['Foot' num2str(k)]), randomized_feature_matrix_test);
acc = sum(guesses == labels_in_test)/length(guesses);
C = confusion.getMatrix([labels_in_test;1; 2; 3], [guesses; 3; 1; 2]);

% correct_vec=[];
% predicted = [false, false];
% guess_vec = [];
% % Fs = 1/(mean(data_fin(2:end,1)-data_fin(1:end-1,1))/1000);
% % figure; hold on; plot(new_time, data_fin(:,2)); plot(new_time, data_fin(:,17)*1000)
% %% Testing
% % figure;
% data_past = [];
% counterL = 0; counterR = 0;
% data_predictL = [];
% data_predictR = [];
% jrange = 1:(two_feet+1):nOutputSignals;
% guess_vec = []; true_labs = [];
% for pt_ind = 1:length(data_test)
%     pt_in= data_test(pt_ind,:);
% %     pt_in(7)=0; pt_in(end)=0;
%     
%     data_predictL = [data_predictL; pt_in];
%     data_predictR = [data_predictR; pt_in];
%     if pt_ind == 1
%         continue
%     end
%     if pt_in(7)==2
%         kk = 1;
%         data_predict = data_predictL;
%     elseif pt_in(end)==2
%         kk = 2;
%         data_predict = data_predictR;
%     else
%         kk = 0;
%     end
%     if kk ~= 0
%         [FootSt_guess] = romanFunction(data_predict);
%         strideList = FootSt_guess.(foot_names{kk});
%         n2= setdiff(foot_names,foot_names{kk});
%         strideListOther = FootSt_guess.(n2{:});
%         if two_feet
%             lenStrideList = min(length(strideList), length(strideListOther));
%         else
%             lenStrideList = length(strideList);
%         end
%         maxs1 = []; means1 = []; mins1 = []; ranges1 = [];
%         iter = 1;
%         for jj=1:length(jrange)
%             jk = jrange(jj);
%             cycle_time = length(strideList.(prediction_signals{iter}));
%             % start after planatar flexion
%             window_start = 1; 
% %             window_end = .6*cycle_time - POST_SWING_CUTOFF_SAMPLES;
%             window_end = cycle_time;
%             means1(jk) = mean(strideList.(prediction_signals{iter})(window_start:window_end));
%             maxs1(jk) = max(strideList.(prediction_signals{iter})(window_start:window_end));
%             mins1(jk) = min(strideList.(prediction_signals{iter})(window_start:window_end));            
%             ranges1(jk) = range(strideList.(prediction_signals{iter})(window_start:window_end));
%             if two_feet
%                 means1(jk+1) = mean(strideListOther.(prediction_signals{iter})(window_start:window_end));
%                 maxs1(jk+1) = max(strideListOther.(prediction_signals{iter})(window_start:window_end));
%                 mins1(jk+1) = min(strideListOther.(prediction_signals{iter})(window_start:window_end));            
%                 ranges1(jk+1) = range(strideListOther.(prediction_signals{iter})(window_start:window_end));
%             end
%             iter = iter+1;
%         end
%         feature_matrix = [maxs1, mins1, ranges1];
%         guesses = predict(Mdl.(['Foot' num2str(k)]), feature_matrix);
%         
%         guess_vec = [guess_vec, guesses];
%         true_labs = [true_labs, labels_test(pt_ind)];
%         if kk == 1
%             data_predictL = [];
%         else
%             data_predictR =[];
%         end
%     end
% 
%     
% %     [data_past, correct_ans, guesses] = pointPredictor3(pt_in, pt_ind, data_past, labels_test(pt_ind), two_feet, cycle_time, Mdl);
% % 
% %     hold on
% %     correct_vec = [correct_vec, correct_ans];
% %     guess_vec = [guess_vec, guesses];
% end
% acc = sum(guess_vec == true_labs)/length(guess_vec);
% C = confusion.getMatrix([true_labs';1; 2; 3], [guess_vec'; 3; 1; 2]);
end