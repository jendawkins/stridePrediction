%% Notes from oct 26 meeting
clear all
% Stride time predictor: 
% - previous (1, 2, 3?) cycle times
% - Ax^2 + Az^2 at HS

POST_SWING_CUTOFF_TIME_S=0.2;
SAMPLE_RATE_HZ = 100;
POST_SWING_CUTOFF_SAMPLES = int32(POST_SWING_CUTOFF_TIME_S * SAMPLE_RATE_HZ);

TWO_FEET = 0;
FOLDS = 5;
PLOT = false;

inverse_kinematics_outputs = {'r0','r1','r2','r3','gAccY','gAccZ','gVelY','gVelZ','gPosY','gPosZ'};
prediction_signals = ['aAccY','aAccZ','aOmegaX','d1aAccY','d1aAccZ','d1aOmegaX','i1aAccY','i1aAccZ','i1aOmegaX',inverse_kinematics_outputs];
foot_names= {'F1','F2'};

dbstop if error
close all
dt = '22-Dec-2018-';
if exist([dt 'processed_data.csv'])==0
    [data_fin] = process_data(pwd);
else
    load([dt 'processed_data.csv'])
    data_fin = X22_Dec_2018_processed_data;
end
new_time = data_fin(:,1);
fin_mat3 = data_fin;
z = find(fin_mat3(:,8)==1);
z2 = find(fin_mat3(:,end-1)==1);

y = z(2:end)-z(1:end-1);
IQR = prctile(y,75)-prctile(y,25);
strides_to_delete = [find(y>prctile(y,75)+IQR*1.5); find(y<prctile(y,25)-IQR*1.5)];
% z = z(y<prctile(y,75)+ IQR*1.5 & y>prctile(y,25)- IQR*1.5);
% y = y(y<prctile(y,75)+ IQR*1.5 & y>prctile(y,25)- IQR*1.5);

y2 = z2(2:end)-z2(1:end-1);
strides_to_delete2 = [find(y2>prctile(y2,75)+IQR*1.5); find(y2<prctile(y2,25)-IQR*1.5)];
% IQR2 = prctile(y2,75)-prctile(y2,25);
% z2 = z2(y2<prctile(y2,75)+ IQR2*1.5 & y2>prctile(y2,25)- IQR*1.5);
% y2 = y2(y2<prctile(y2,75)+ IQR2*1.5 & y2>prctile(y2,25)- IQR*1.5);

stPF = round(z(1:end-1) +.3.*y -15); edPF = round(z(1:end-1) + .6*y);
stPF2 = round(z2(1:end-1) +.3*y2 -15); edPF2 = round(z2(1:end-1) + .6*y2);
labels = fin_mat3(:,end);
data_in = fin_mat3(:,[2:15]);

% Calculate angels from data???

% data_in(:,[1:6,8:13]) = (data_in(:,[1:6,8:13]) - mean(data_in(:,[1:6,8:13])))./std(data_in(:,[1:6,8:13]));

data_in(stPF,7)=2; data_in(edPF,7)=3;
data_in(stPF2,14)=2; data_in(edPF2,14)=3;

% for iters = 1:10
%     TWO_FEET = mod(
STRIDE_MARKER = 1;
iter = 1;
figure; 
for tf = 1:2
    TWO_FEET = tf-1;
    accvec = [];
for ft = 1:2 % foot 1 or 2
%     stMark1 = z;
%     stMark2 = z2;
    stMark1 = find(data_in(:,7)==STRIDE_MARKER);
    stMark2 = find(data_in(:,14)==STRIDE_MARKER);
    if ft==2
%         stMark1 = z2;
%         stMark2 = z;
        stMark1 = stMark2;
        stMark2 = stMark1;
        strides_to_delete = strides_to_delete2;
    end
%     stridesAll = FootStrAll.([foot_names{ft}]);
%     stridesAllOther = FootStrAll.([foot_names{setdiff(1:2,ft)}]);
    
    select_labels = labels(stMark1);
    last_lab = select_labels(end);
    select_labels = select_labels(2:end); %***have labels as next startPF
    cvIndices = crossvalind('Kfold',length(stMark1)-1,FOLDS, 'Min',3);
    
    for cv = 1:FOLDS
        sample_strides = find(any(cvIndices == setdiff(1:FOLDS,cv),2));
        sample_strides = setdiff(sample_strides, strides_to_delete);
        test_strides = find(cvIndices == cv);
        test_strides = setdiff(test_strides, strides_to_delete);
        
        % split training testing
        data_train = [];
        for i = 1:length(sample_strides)
            ii = sample_strides(i);
            data_train = [data_train; data_in(stMark1(ii):stMark1(ii+1)-1,:)];
        end
        
        HStrain = [find(data_train(:,ft*7)==1); size(data_train,1)+1];
        trainCycles = HStrain(2:end)-HStrain(1:end-1);
        LRlabels_tr = [];
        for tc = 1:length(trainCycles)
            LRlabels_tr = [LRlabels_tr linspace(0,trainCycles(tc)*SAMPLE_RATE_HZ...
                ,trainCycles(tc))./(trainCycles(tc)*SAMPLE_RATE_HZ)];
        end
        
        if TWO_FEET
            feature_mat = data_train(:,[1:6,8:13]);
        else
            feature_mat = data_train(:,1:6);
        end
        
        % Ridge Regression
%         X = data_train;
%         y = LRlabels_tr';
%         D = x2fx(X,'interaction');
%         D(:,1) = []; % No constant term
%         k = 0:1e-5:5e-3;
%         b = ridge(y,D,k);
%         
%         figure
%         plot(k,b,'LineWidth',2)
%         ylim([-100 100])
%         grid on
%         xlabel('Ridge Parameter')
%         ylabel('Standardized Coefficient')
%         title('{\bf Ridge Trace}')
%         legend('x1','x2','x3')

        b = regress(LRlabels_tr',feature_mat);
        
        data_test = [];
        labtest2 = [];
        for j = 1:length(test_strides)
            jj = test_strides(j);
            data_test = [data_test; data_in(stMark1(jj):stMark1(jj+1)-1,:)];
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

        
        % Testing
        guess_vec = [];
        data_pred = [];
        mse_loss = 0;
        if cv == 1
%             subplot(2,2,iter)
            figure
            iter = iter+1;
            plot([0 0],[1, 1]);
            hold on;
            xlabel('Actual cycle time')
            ylabel('Predicted cycle time')
            axis([0 1 0 1])
        end
        pl = 1;
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
            pred_time = pt_in*b;
            mse_loss = mse_loss + (LRlabels_ts(pt_idx) - pred_time).^2;
            if cv==1
                scatter(LRlabels_ts(pt_idx), pred_time, 10, col,'Filled');      
            end
        end
%         legend(p)
        tvec = {'One Foot Training','Two Feet Training'};
        title(['Fold ' num2str(cv), ', Foot ' num2str(ft) ', ' tvec{tf}])
        mse_loss_t = mse_loss / size(data_test,1);
    end
  
end

end

