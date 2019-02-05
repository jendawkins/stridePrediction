%% Notes from oct 26 meeting
clear all
% CROSSVAL in MATLAB
% ANOVA in statistical significance 
% Figure of One leg vs Two Legs!!!
% one leg vs two legs, window vs stride
% try to make bar graph next week!!!

% Use least squares regression for predicting time
% inputs are continuous features, output is number = ms to stance -- linear
% regression??

% if two legs is better, how early to make prediction??
plot_pts = 0;
POST_SWING_CUTOFF_TIME_S=0.2;
SAMPLE_RATE_HZ = 100;
POST_SWING_CUTOFF_SAMPLES = int32(POST_SWING_CUTOFF_TIME_S * SAMPLE_RATE_HZ);

TWO_FEET = 0;
FOLDS = 5;

% inverse_kinematics_outputs = {'r0','r1','r2','r3','gAccY','gAccZ','gVelY','gVelZ','gPosY','gPosZ'};
% prediction_signals = ['aAccY','aAccZ','aOmegaX','d1aAccY','d1aAccZ','d1aOmegaX','i1aAccY','i1aAccZ','i1aOmegaX',inverse_kinematics_outputs];

% 
prediction_signals = {'aAccX','aAccY','aAccZ','d1aOmegaX','d1aOmegaY','d1aOmegaZ','d1aAccX','d1aAccY','d1aAccZ'};
foot_names= {'F1','F2'};

dt = '22-Dec-2018';
dbstop if error
close all
if exist([dt '-processed_data.csv'])==0
    [data_fin] = process_data(pwd);
else
    load([dt '-processed_data.csv'])
    data_fin = X22_Dec_2018_processed_data;
end
new_time = data_fin(:,1);
fin_mat3 = data_fin;
z = find(fin_mat3(:,8)==1);
z2 = find(fin_mat3(:,end-1)==1);

% r2 = sqrt(data_fin(:,2).^2 + data_fin(:,3).^2);
% r20 = sqrt(data_fin(:,9).^2 + data_fin(:,10).^2);
% plot(data_fin(:,1), r2,'m');
% hold on
% scatter(data_fin(z,1), zeros([1,length(z)]),'r*');
% plot(data_fin(:,1), r20,'b');
% scatter(data_fin(z2,1), zeros([1,length(z2)]),'g*');


% fix discrepancy between feet
% Problem: I can use two feet like I do below, but predictions are pretty
% bad since I'm calculating a random window. Need to use prior foot's swing

y = z(2:end)-z(1:end-1);
IQR = prctile(y,75)-prctile(y,25);
% z = z(y<prctile(y,75)+ IQR*1.5 & y>prctile(y,25)- IQR*1.5);
% y = y(y<prctile(y,75)+ IQR*1.5 & y>prctile(y,25)- IQR*1.5);

y2 = z2(2:end)-z2(1:end-1);
% IQR2 = prctile(y2,75)-prctile(y2,25);
% z2 = z2(y2<prctile(y2,75)+ IQR2*1.5 & y2>prctile(y2,25)- IQR*1.5);
% y2 = y2(y2<prctile(y2,75)+ IQR2*1.5 & y2>prctile(y2,25)- IQR*1.5);

labels1o = fin_mat3(:,9);
labels1 = labels1o(z);
labels2o = fin_mat3(:,end);
labels2 = labels2o(z2);

delstr1 = find(y<prctile(y,25)-IQR*1.5);
delstr2 = find(y2<prctile(y2,25)-IQR*1.5);

z2(delstr2)=[];
z(delstr1)=[];
y(delstr1) = [];
y2(delstr2)=[];
labels1(delstr1) = [];
labels2(delstr2) = [];

zest2 = round(z(1:end-1) + .6*y);
zest1 = round(z2(1:end-1) + .6*y2);

z = [z(1); zest2(1); z(2:end)];
labels1 = [labels1(1); labels2o(zest2(1)); labels1(2:end)];
for j = 2:(length(z)-1)
    if z2(j)>z(j) && z2(j+1)<z(j+1) % gap in green
%         continue
%     else
        labest = mean([labels2o(z2(j)) labels2o(z2(j+1))]);
        if isinteger(labest)==0
            labest = 0;
        end
        zest = round(z2(j) + (z2(j+1)-z(j))*.6);
        z = [z(1:j); zest; z(j+1:end)];
        labels1 = [labels1(1:j); labest; labels1(j+1:end)];
%         z2 = [z2(1:j); zest2(j); z2(j+1:end)];
    end
    % gap in red
    if z(j)>z2(j-1) && z(j+1)<z2(j)
%         continue
%     else
        labest = mean([labels1o(z(j)) labels1o(z(j+1))]);
        if isinteger(labest)==0
            labest = 0;
        end
        zest = round(z(j) + (z(j+1)-z(j))*.6);
        z2 = [z2(1:j-1); zest; z2(j:end)];
        labels2 = [labels2(1:j-1); labest; labels2(j:end)];
    end
end

y1 = z(2:end)-z(1:end-1);
y2 = z2(2:end)-z2(1:end-1);

delstr1 = find(y1<prctile(y,25)-IQR*1.5);
delstr2 = find(y2<prctile(y2,25)-IQR*1.5);

labels1([delstr1; delstr2]) = [];
labels2([delstr1; delstr2]) = [];

z([delstr1; delstr2]) = [];
z2([delstr1; delstr2]) = [];

data_in = fin_mat3(:,[2:8,10:16]);

data_in2 = data_in;
data_in2(:,7) = zeros(size(data_in2(:,7)));
data_in2(:,end) = zeros(size(data_in2(:,7)));

% just changed this but didnt work -- start here
% zerolabs = [find(labels1==0)-1; find(labels2==0)-1];
% z(zerolabs)=[];
% z2(zerolabs) = [];

data_in2(z,7) = ones(size(z));
data_in2(z2,end) = ones(size(z2));

y = z(2:end) - z(1:end-1);
y2 = z2(2:end) - z2(1:end-1);

stPF = round(z(1:end-1) +.3.*y -15); edPF = round(z(1:end-1) + .6*y);
stPF2 = round(z2(1:end-1) +.3*y2 -15); edPF2 = round(z2(1:end-1) + .6*y2);

strides_to_delete1 = [find(y>prctile(y,75)+IQR*2); find(y<prctile(y,25)-IQR*1.5)];
strides_to_delete2 = [find(y2>prctile(y2,75)+IQR*2); find(y2<prctile(y2,25)-IQR*1.5)];

strides_to_delete1 = [strides_to_delete1; find(labels1==0)-1];
strides_to_delete2 = [strides_to_delete2; find(labels2==0)-1];

data_in = data_in2;
% Calculate angels from data???

data_in(stPF,7)=2; data_in(edPF,7)=3;
data_in(stPF2,14)=2; data_in(edPF2,14)=3;

% for iters = 1:10
%     TWO_FEET = mod(
STRIDE_MARKER = 1;
NUM_INPUT_TIMES = 11;
% mse_tot = zeros(2);
for tf = 1:2
    TWO_FEET = tf-1;
    accvec = [];
    
    
    layers = [ ...
        imageInputLayer([NUM_INPUT_TIMES (tf*6) 1]) % 4x6x1 --> 3x4x3 = 36
        convolution2dLayer([2 3],3)
        reluLayer
        fullyConnectedLayer(4)
        regressionLayer];

    for ft = 1:2 % foot 1 or 2
        %     stMark1 = z;
        %     stMark2 = z2;
        stMark1 = find(data_in(:,7)==STRIDE_MARKER);
        stMark2 = find(data_in(:,14)==STRIDE_MARKER);
        select_labels = labels1;
        strides_to_delete = strides_to_delete1;
        if ft==2
            %         stMark1 = z2;
            %         stMark2 = z;
            stMark1 = stMark2;
            stMark2 = stMark1;
            strides_to_delete = strides_to_delete2;
            select_labels = labels2;
        end
        %     stridesAll = FootStrAll.([foot_names{ft}]);
        %     stridesAllOther = FootStrAll.([foot_names{setdiff(1:2,ft)}]);
        
        last_lab = select_labels(end);
        select_labels = select_labels(2:end); %***have labels as next startPF
        cvIndices = crossvalind('Kfold',min([length(stMark1), length(stMark2)])-2,FOLDS, 'Min',3);
        
        cv = 1;
        while cv <= FOLDS
            sample_strides = find(any(cvIndices == setdiff(1:FOLDS,cv),2));
            sample_strides = setdiff(sample_strides, strides_to_delete);
            test_strides = find(cvIndices == cv);
            test_strides = setdiff(test_strides, strides_to_delete);
            
            % delete extra stride 1
            str1 = find(select_labels(sample_strides)==1);
            num2del = length(str1) - length(find(select_labels(sample_strides)==2));
            sample_strides = setdiff(sample_strides, sample_strides(str1(randsample(length(str1),num2del))));
            if all(unique(select_labels(sample_strides)) ~= [1;2;3]) || all(unique(select_labels(test_strides)) ~= [1;2;3])
                continue
            end
            
            tslab = select_labels(test_strides);
            trlab = select_labels(sample_strides);
            
            % split training testing
            data_train = [];
            labtrain2 = [];
            for i = 1:length(sample_strides)
                ii = sample_strides(i);
                data_train = [data_train; data_in(stMark1(ii):stMark1(ii+1)-1,:)];
                labtrain2 = [labtrain2; repmat(trlab(i),[length(stMark1(ii):stMark1(ii+1)-1),1])];
            end
            
            data_test = [];
            labtest2 = [];
            for j = 1:length(test_strides)
                jj = test_strides(j);
                data_test = [data_test; data_in(stMark1(jj):stMark1(jj+1)-1,:)];
                labtest2 = [labtest2; repmat(tslab(j),[length(stMark1(jj):stMark1(jj+1)-1),1])];
            end
            % add last stride to data_test
            %         data_test = [data_test; data_in(stMark1(jj+1):end,:)];
            
            %         tslab = [select_labels(test_strides); last_lab];
            

            % train
            tr_strides = [find(data_train(:,7*ft)==STRIDE_MARKER); length(data_train)];
            ts_strides = [find(data_test(:,7*ft)==STRIDE_MARKER); length(data_test)];
%             tr_strides = stMark1(sample_strides);
%             cycle_time_train = y(sample_strides);
            
            HStrain = [find(data_train(:,ft*7)==1); size(data_train,1)+1];
            trainCycles = HStrain(2:end)-HStrain(1:end-1);
            LRlabels_tr = [];
            for tc = 1:length(trainCycles)
                LRlabels_tr = [LRlabels_tr linspace(0,trainCycles(tc)*SAMPLE_RATE_HZ...
                    ,trainCycles(tc))./(trainCycles(tc)*SAMPLE_RATE_HZ)];
            end
            
            if TWO_FEET
                data_train = data_train(:,[1:6,8:13]);
            else
                data_train = data_train(:,[1:6]);
            end            
            
            ftmat_train = zeros([NUM_INPUT_TIMES, size(data_train,2), 1, length(data_train)-NUM_INPUT_TIMES]);
            labels_tr_fin = [];
            labtrain22 = [];
            for rs = 1:size(ftmat_train,4)
                ftmat_train(:,:,:,rs) = data_train(rs:rs+NUM_INPUT_TIMES-1,:);
                labels_tr_fin(rs) = median(LRlabels_tr(rs:rs+NUM_INPUT_TIMES-1));
                labtrain22(rs) = median(labtrain2(rs:rs+NUM_INPUT_TIMES-1));
            end

            rperm = randperm(size(ftmat_train,4));
            ftmat_train = ftmat_train(:,:,:,rperm);
            labels_tr_fin = labels_tr_fin(rperm);
            
            % make validation data (i.e. test data)          
            HStest = [find(data_test(:,ft*7)==1); size(data_test,1)+1];
            testCycles = HStest(2:end)-HStest(1:end-1);
            LRlabels_ts = [];
            for tc = 1:length(testCycles)
                LRlabels_ts = [LRlabels_ts linspace(0,testCycles(tc)*SAMPLE_RATE_HZ...
                    ,testCycles(tc))./(testCycles(tc)*SAMPLE_RATE_HZ)];
            end
            
           if TWO_FEET
                data_test = data_test(:,[1:6,8:13]);
            else
                data_test = data_test(:,[1:6,]);
            end     
            
            ftmat_test = zeros([NUM_INPUT_TIMES, size(data_test,2), 1, length(data_test)-NUM_INPUT_TIMES]);
            labels_ts_fin = [];
            labtest22 = [];
            for rs = 1:length(data_test)-NUM_INPUT_TIMES
                ftmat_test(:,:,:,rs) = data_test(rs:rs+NUM_INPUT_TIMES-1,:);
                labels_ts_fin(rs) = median(LRlabels_ts(rs:rs+NUM_INPUT_TIMES-1));
                labtest22(rs) = median(labtest2(rs:rs+NUM_INPUT_TIMES-1));
            end
            
            labtest22_OH = make_one_hot(round(labtest22)',[1,2,3]);
            options = trainingOptions('rmsprop', ...
                'MaxEpochs',10, ...
                'Shuffle','every-epoch', ...
                'ValidationFrequency',30, ...
                'Verbose',false, ...
                'ValidationData',{ftmat_test, [labels_ts_fin' labtest22_OH]});
            labtrain22_OH = make_one_hot(round(labtrain22)',[1,2,3]);
            net = trainNetwork(ftmat_train,[labels_tr_fin' labtrain22_OH],layers,options);
            
            % test non-rolling
            
            %% start here -- make labels one hot
            guess = predict(net, ftmat_test);
            time_guess = guess(:,1);
            label_guess = guess(:,2:end);
            [probs, guesses] = max(exp(label_guess)./sum(exp(label_guess),2),[],2);
%             [probs2, guesses2] = max(softmax(label_guess),[],2);
%             [probs, guesses] = max(predict(net, ftmat_test)');

            accTest = sum(guesses==round(labtest22)')/length(guesses);
            CTest = confusion.getMatrix([round(labtest22)';1; 2; 3], [guesses; 3; 1; 2],0);
            disp('static predictions')
            disp(accTest)
            disp(['Fold ' num2str(cv)])
            disp(CTest)
            
            mse_loss(cv) = sum((labels_ts_fin' - time_guess).^2)/length(time_guess);
            if ft == 1
                figure(tf)
                tvec = {'One Foot Training','Two Feet Training'};
                plot([0,1],[0,1],'k','LineWidth',2)
                hold on
                gscatter(labels_ts_fin', time_guess, round(labtest22));  
                xlabel('Actual cycle time')
                ylabel('Predicted cycle time')
                title([tvec{tf}])
                axis([0 1 -inf inf])
            end
            
            %% this part commented out for now
            % Testing
%             guesses = []; time_guess = [];
%             start = 1;
%             for pt = 1:size(ftmat_test,4)
%                 guess_1 = predict(net, ftmat_test(:,:,:,pt));
%                 time_guess_1 = guess_1(:,1);
%                 label_guess_1 = guess_1(:,2:end);
%                 [probs_1, guesses_1] = max(exp(label_guess_1)./sum(exp(label_guess_1),2),[],2);
% %             [probs2, guesses2] = max(softmax(label_guess),[],2);
% %             [probs, guesses] = max(predict(net, ftmat_test)');
%                 guesses(pt) = guesses_1;
%                 time_guess(pt) = time_guess_1;
%                 col = {'r','g','b'};
%                 if ft == 1 && cv == 1 && (pt==size(ftmat_test,4) || (labels_ts_fin(pt)>.9 && labels_ts_fin(pt+1)<.1))
%                     figure(tf)
%                     endpt = pt;
%                     plot_label = round(mean(labtest22(start:endpt)));
%                     tvec = {'One Foot Training','Two Feet Training'};
%                     plot([0,1],[0,1])
%                     hold on
%                     scatter(labels_ts_fin(start:endpt)', time_guess(start:endpt), 10, 'Filled', col{plot_label});  
%                     xlabel('Actual cycle time')
%                     ylabel('Predicted cycle time')
%                     title([tvec{tf}])
%                     axis([0 1 -inf inf])  
%                     time_guess_plot = [];
%                     start = endpt+1;
%                 end
% 
%             end
%             p = findobj(gca,'Type','Patch');
%             legend(p(I),'Upstairs','Downstairs','Flat')
%             accTest = sum(guesses==round(labtest22)')/length(guesses);
%             CTest = confusion.getMatrix([round(labtest22)';1; 2; 3], [guesses; 3; 1; 2],0);
%             disp('static predictions')
%             disp(accTest)
%             disp(['Fold ' num2str(cv)])
%             disp(CTest)
            
%             if ft == 1
%                 figure(tf)
%                 tvec = {'One Foot Training','Two Feet Training'};
%                 plot([0,1],[0,1])
%                 hold on
%                 scatter(labels_ts_fin', time_guess, 10, 'Filled');  
%                 xlabel('Actual cycle time')
%                 ylabel('Predicted cycle time')
%                 title([tvec{tf}])
%                 axis([0 1 -inf inf])
%             end
%             guess_vec = [];
%             data_pred = [];
% %             if plot_pts
% %                 figure;
% %                 title('Rolling Predictions 2')
% %             end
%             labiter = 1;
%             start = 1;
%             gp_mat = {'r','g','b'};
%             gp_matOther = {'m','c','y'};
%             for pt_idx = 1:size(data_test,1)
%                 pt_in = data_test(pt_idx,:);
%                 stride_check = ft*7;
%                 data_pred = [data_pred; pt_in];
%                 if pt_idx == 1
%                     continue
%                 end
%                 
%                 
%                 if pt_in(stride_check)==STRIDE_MARKER || pt_idx == size(data_test,1)
%                     
%                     FtMat = romanFunction(data_pred(1:end-1,:));
%                     stridesTest = FtMat.([foot_names{ft}]);
%                     stridesTestOther = FtMat.([foot_names{setdiff(1:2,ft)}]);
%                     ftmat_test = createFeatureMatrix2(stridesTest, stridesTestOther, prediction_signals, TWO_FEET);
%                     
%                     guess = predict(Mdl, ftmat_test);
%                     data_pred = [pt_in];
%                     guess_vec = [guess_vec; guess];
%                     if all(ftmat_test ~= ftmat_test_nr(labiter,:))
%                         disp('unequal matrices')
%                     end
%                     if plot_pts
%                         local_t = start:length(stridesTest.a1Raw)+start-1;
%                         local_tOther = start:length(stridesTestOther.a1Raw)+start-1;
%                         for iter =1:length(prediction_signals)
%                             figure(iter);
%                             hold on;
%                             window_start = round(.6*length(stridesTest.a1Raw));
%                             window_end = window_start + 10;
%                             plot(local_t(window_start:window_end),stridesTest.(prediction_signals{iter})(window_start:window_end),gp_mat{tslab(labiter)})
%                             if TWO_FEET
%                                 window_start0 = 30;
%                                 window_end0 = window_start0 + 10;
%                                 plot(local_tOther(window_start0:window_end0),stridesTestOther.(prediction_signals{iter})(window_start0:window_end0),[gp_mat{tslab(labiter)} '--'])
%                                 legend('Foot Pred','Foot Other')
%                             end
%                             title(prediction_signals{iter})
%                             hold on;
%                             ax = gca;
%                         end
%                     end
%                     start = length(stridesTest.a1Raw)+start;
%                     labiter = labiter+1;
%                     
%                 end
%             end
%             acc = sum(guess_vec==tslab)/length(guess_vec);
%             C = confusion.getMatrix([tslab;1; 2; 3], [guess_vec; 3; 1; 2],0);
%             disp('Rolling predictions:')
%             disp(acc)
%             disp(['Fold ' num2str(cv)])
%             disp(C)
            
            
            accv(cv) = accTest;
            cv = cv+1;
        end
        mse_tot(ft,:) = mse_loss;
        Cstruct.(['Foot' num2str(ft)]).(['Fold' num2str(cv)]) = CTest;
        accvec.(['Foot' num2str(ft)]) = accv;
        
    end
    figure(3);
    subplot(1,2,tf)
    tvec = {'One Foot Training','Two Feet Training'};
    
    plot(accvec.Foot1)
    hold on
    plot(accvec.Foot2)
    title(tvec{tf})
    axis([-inf inf 0 1])
    xlabel('Cross Val Folds')
    ylabel('Testing Accuracy')
    legend('Foot L', 'Foot R')
    xticklabels(1:FOLDS)
    
    
    figure(4);
    ax(tf) = subplot(1,2,tf);
    tvec = {'One Foot Training','Two Feet Training'};
    
    plot(mse_tot(1,:))
    hold on
    plot(mse_tot(2,:))
    title(tvec{tf})
    axis([-inf inf -inf inf])
    xlabel('Cross Val Folds')
    ylabel('MSE Loss')
    legend('Foot L', 'Foot R')
    xticklabels(1:FOLDS)
    mse_tot_tot(tf,:,:) = mse_tot;
end
linkaxes([ax(1),ax(2)],'xy')
%  set(ax(1), 'Position', [1 5 0 0.07]);
%  set(ax(2), 'Position', get(ax(1), 'Position'));
%% plot data
figure;
plot(data_in2(:,1))
hold on
scatter(find(data_in2(:,7)==1), [labels1(2:end); 0].*ones([length(find(data_in2(:,7)==1)),1]),'Filled','r'); hold on; 
scatter(find(data_in2(:,end)==1),[labels2(2:end); 0].*ones([length(find(data_in2(:,end)==1)),1]),'Filled','g')
stMark1 = find(data_in2(:,7)==1);
stMark2 = find(data_in2(:,end)==1);

for i = 1:length(stMark1)-1
    if any(i==strides_to_delete1)
        continue
    else
        hold on
        plot(stMark1(i):stMark1(i+1), labels1(i+1)*ones([length(stMark1(i):stMark1(i+1)),1]),'-r');
    end
end

for i = 1:length(stMark2)-1
    if any(i==strides_to_delete2)
        continue
    else
        hold on
        plot(stMark2(i):stMark2(i+1), labels2(i+1)*ones([length(stMark2(i):stMark2(i+1)),1]),'-g');
    end
end
    
    