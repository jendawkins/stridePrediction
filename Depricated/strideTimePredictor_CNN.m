%% Notes from oct 26 meeting
clear all
% Stride time predictor: 
% - previous (1, 2, 3?) cycle times
% - Ax^2 + Az^2 at HS

POST_SWING_CUTOFF_TIME_S=0.2;
SAMPLE_RATE_HZ = 100;
POST_SWING_CUTOFF_SAMPLES = int32(POST_SWING_CUTOFF_TIME_S * SAMPLE_RATE_HZ);
NUM_INPUT_TIMES= 4;
plot_pts = 0;
POST_SWING_CUTOFF_TIME_S=0.2;
SAMPLE_RATE_HZ = 100;
POST_SWING_CUTOFF_SAMPLES = int32(POST_SWING_CUTOFF_TIME_S * SAMPLE_RATE_HZ);

TWO_FEET = 0;
FOLDS = 5;
SYNC_STRIDES = 1;
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

y = z(2:end)-z(1:end-1);
IQR = prctile(y,75)-prctile(y,25);
% z = z(y<prctile(y,75)+ IQR*1.5 & y>prctile(y,25)- IQR*1.5);
% y = y(y<prctile(y,75)+ IQR*1.5 & y>prctile(y,25)- IQR*1.5);

y2 = z2(2:end)-z2(1:end-1);
IQR2 = prctile(y2,75)-prctile(y2,25);
% z2 = z2(y2<prctile(y2,75)+ IQR2*1.5 & y2>prctile(y2,25)- IQR*1.5);
% y2 = y2(y2<prctile(y2,75)+ IQR2*1.5 & y2>prctile(y2,25)- IQR*1.5);

labels1o = fin_mat3(:,9);
labels1 = labels1o(z);
labels2o = fin_mat3(:,end);
labels2 = labels2o(z2);

if SYNC_STRIDES
    delstr1 = find(y<prctile(y,25)-IQR*1.5);
    delstr2 = find(y2<prctile(y2,25)-IQR2*1.5);
    
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
    data_in = data_in2;
end

stPF = round(z(1:end-1) +.3.*y -15); edPF = round(z(1:end-1) + .6*y);
stPF2 = round(z2(1:end-1) +.3*y2 -15); edPF2 = round(z2(1:end-1) + .6*y2);

strides_to_delete1 = [find(y>prctile(y,75)+IQR*2); find(y<prctile(y,25)-IQR*1.5)];
strides_to_delete2 = [find(y2>prctile(y2,75)+IQR*2); find(y2<prctile(y2,25)-IQR*1.5)];

strides_to_delete1 = [strides_to_delete1; find(labels1==0)-1];
strides_to_delete2 = [strides_to_delete2; find(labels2==0)-1];
% Calculate angels from data???

data_in(stPF,7)=2; data_in(edPF,7)=3;
data_in(stPF2,14)=2; data_in(edPF2,14)=3;

% for iters = 1:10
%     TWO_FEET = mod(
STRIDE_MARKER = 1;
% NUM_INPUT_TIMES = 10;

% for iters = 1:10
%     TWO_FEET = mod(


iter = 1;
for tf = 1:2
    TWO_FEET = tf-1;
    accvec = [];
    for ft = 1:2 % foot 1 or 2
        
        layers = [ ...
            imageInputLayer([NUM_INPUT_TIMES (tf*6) 1]) % 4x6x1 --> 3x4x3 = 36
            convolution2dLayer([2 3],3)
            reluLayer
            fullyConnectedLayer(1)
            regressionLayer];

%         layers(2,1).Bias = rand(1,1,3);
%         layers(2,1).Weights = rand(2,3,1,3);
%         layers(4,1).Weights = rand(1,36);
        
%         lgraph = layerGraph(layers);
        %     stMark1 = z;
%     stMark2 = z2;
    stMark1 = find(data_in(:,7)==STRIDE_MARKER);
    stMark2 = find(data_in(:,14)==STRIDE_MARKER);
    strides_to_delete = strides_to_delete1;
    select_labels = labels1;
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
    
%     select_labels = labels(stMark1);
    last_lab = select_labels(end);
    select_labels = select_labels(2:end); %***have labels as next startPF
    cvIndices = crossvalind('Kfold',length(stMark1)-1,FOLDS, 'Min',3);
    
    for cv = 1:FOLDS
        sample_strides = find(any(cvIndices == setdiff(1:FOLDS,cv),2));
%         sample_strides = setdiff(sample_strides, strides_to_delete);
        test_strides = find(cvIndices == cv);
%         test_strides = setdiff(test_strides, strides_to_delete);
        tslab = select_labels(test_strides);
        % split training testing
        data_train = [];
        for i = 1:length(sample_strides)
            ii = sample_strides(i);
            data_train = [data_train; data_in(stMark1(ii):stMark1(ii+1)-1,:)];
        end
        
        HStrain = [find(data_train(:,ft*7)==1); size(data_train,1)+1];
        trainCycles = HStrain(2:end)-HStrain(1:end-1);
        LRlabels_tr = [];
        cycle_time = [mean(trainCycles); trainCycles];
        cycle_time_long = [];
        for tc = 1:length(trainCycles)
            LRlabels_tr = [LRlabels_tr linspace(0,trainCycles(tc)*SAMPLE_RATE_HZ...
                ,trainCycles(tc))./(trainCycles(tc)*SAMPLE_RATE_HZ)];
            cycle_time_long = [cycle_time_long; repmat(cycle_time(tc), [cycle_time(tc+1),1])];
        end
        
        if TWO_FEET
            feature_mat = data_train(:,[1:6,8:13]);
        else
            feature_mat = data_train(:,1:6);
        end
        
%         feature_mat = feature_mat.*cycle_time_long;
        feature_mat = [feature_mat; zeros(mod(-size(feature_mat,1),NUM_INPUT_TIMES), size(feature_mat,2))];
        feature_NN = reshape(feature_mat,[NUM_INPUT_TIMES, size(feature_mat,2), 1, size(feature_mat,1)/NUM_INPUT_TIMES]);
        LRlabels_tr = [LRlabels_tr zeros(1,mod(-length(LRlabels_tr),NUM_INPUT_TIMES))];
        label_idx = 1:NUM_INPUT_TIMES:size(feature_mat,1);
        LRlabels_tr = LRlabels_tr(label_idx);
        
        %        b = regress(LRlabels_tr',feature_mat);
%         net = trainNetwork(feature_NN,LRlabels_tr',layers,options);
        
        data_test = [];
        labtest2 = [];
        for j = 1:length(test_strides)
            jj = test_strides(j);
            data_test = [data_test; data_in(stMark1(jj):stMark1(jj+1)-1,:)];
            labtest2 = [labtest2; repmat(tslab(j),[length(stMark1(jj):stMark1(jj+1)-1),1])];
        end
        % add last stride to data_test
%         data_test = [data_test; data_in(stMark1(jj+1):end,:)];

        HStest = [find(data_test(:,ft*7)==1); size(data_test,1)+1];
        testCycles = HStest(2:end)-HStest(1:end-1);
        LRlabels_ts = [];
        for tc = 1:length(testCycles)
            LRlabels_ts = [LRlabels_ts linspace(0,testCycles(tc)*SAMPLE_RATE_HZ...
                ,testCycles(tc))./(testCycles(tc)*SAMPLE_RATE_HZ)];
        end
        LRlabels_ts = [LRlabels_ts zeros(1,mod(-length(LRlabels_ts),NUM_INPUT_TIMES))];
        label_idx = 1:NUM_INPUT_TIMES:length(LRlabels_ts);
        LRlabels_ts4 = LRlabels_ts(label_idx);
        labtest_terrain = labtest2(label_idx);
        
        if TWO_FEET
            feature_val = data_test(:,[1:6,8:13]);
        else
            feature_val = data_test(:,1:6);
        end
        
        feature_val = [feature_val; zeros(mod(-size(feature_val,1),NUM_INPUT_TIMES), size(feature_val,2))];
        feature_NN_val = reshape(feature_val,[NUM_INPUT_TIMES, size(feature_val,2), 1, size(feature_val,1)/NUM_INPUT_TIMES]);
%                     'ValidationData',{data_test,LRlabels_ts4'},...

        options = trainingOptions('rmsprop', ...
            'MaxEpochs',15, ...
            'Shuffle','every-epoch', ...
            'ValidationFrequency',30, ...
            'Verbose',false, ...
            'ValidationData',{feature_NN_val,LRlabels_ts4'}, ...
            'Plots','training-progress');
        
        net = trainNetwork(feature_NN,LRlabels_tr',layers,options);
        
        % Testing
        guess_vec = [];
        data_pred = [];
        mse_loss = 0;
%         if cv == 1
% %             subplot(2,2,iter)
%             figure
%             iter = iter+1;
%             plot([0 1],[0 1]);
%             hold on;
%             xlabel('Actual cycle time')
%             ylabel('Predicted cycle time')
%             axis([0 1 -inf inf])
%         end
        pl = 1;
        pts_pred = [];
        pred_times = [];
        labels_fin = [];
        tst_cycles = [mean(testCycles); testCycles];
        iter = 1;
        for pt_idx = 1:size(data_test,1)
            if LRlabels_ts(pt_idx)== 0
                col = rand(1,3);
                pl = pl+1;
            end

            if TWO_FEET
                pt_in = data_test(pt_idx,[1:6,8:13]);
            else
                pt_in = data_test(pt_idx,[1:6]);
            end
            pts_pred = [pts_pred; pt_in];
            if size(pts_pred,1)>=4
%                 pts_pred = pts_pred*tst_cycles(pl);
                pred_time = predict(net, pts_pred);
                pts_pred = [];
                mse_loss = mse_loss + (LRlabels_ts(iter) - pred_time).^2;
%                 if cv==1
%                     scatter(LRlabels_ts(pt_idx), pred_time, 10, col);      
%                 end
                pred_times = [pred_times; pred_time];
                labels_fin = [labels_fin; LRlabels_ts4(iter)];
                iter = iter+1;
            end
        end
%         legend(p)
        if cv == 1
            gps = {'Upstairs','Downstairs','Level Ground'};
            z_labels = find(labtest_terrain==0);
            pred_times(z_labels) = []; labels_fin(z_labels) = [];
            labtest_terrain(z_labels) = [];
            figure(tf)
            tvec = {'One Foot Training','Two Feet Training'};
            plot([0,1],[0,1],'k','LineWidth',2)
            hold on
            gscatter(labels_fin, pred_times, gps(round(labtest_terrain(1:length(labels_fin))))');  
            xlabel('Actual cycle time')
            ylabel('Predicted cycle time')
            title([tvec{tf}])
            axis([0 1 -inf inf])
        end
%         tvec = {'One Foot Training','Two Feet Training'};
%         title(['Fold ' num2str(cv), ', Foot ' num2str(ft) ', ' tvec{tf}])
        mse_loss_t(cv) = mse_loss / size(data_test,1);
    end
    mse_tot(ft,:) = mse_loss_t;
  
    end
    figure(3);
    ax(tf) = subplot(1,2,tf);
    tvec = {'One Foot Training','Two Feet Training'};
    
    plot(mse_tot(1,:))
    hold on
    plot(mse_tot(2,:))
    title(tvec{tf})
%     axis([-inf inf -inf inf])
    xlabel('Cross Val Folds')
    ylabel('MSE Loss')
    legend('Foot L', 'Foot R')
    xticklabels(1:FOLDS)

end
linkaxes([ax(1),ax(2)],'xy')
