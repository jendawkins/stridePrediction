%% Notes from oct 26 meeting
clear all
% Stride time predictor: 
% - previous (1, 2, 3?) cycle times
% - Ax^2 + Az^2 at HS

% ridge regression

POST_SWING_CUTOFF_TIME_S=0.2;
SAMPLE_RATE_HZ = 100;
POST_SWING_CUTOFF_SAMPLES = int32(POST_SWING_CUTOFF_TIME_S * SAMPLE_RATE_HZ);

TWO_FEET = 0;
FOLDS = 5;
PLOT = false;

raw_sensor_outputs = {'a1Raw','a2Raw','a3Raw','g1Raw','g2Raw','g3Raw'};
inverse_kinematics_outputs = {'r0','r1','r2','r3','gAccY','gAccZ','gVelY','gVelZ','gPosY','gPosZ'};
prediction_signals = ['aAccY','aAccZ','aOmegaX','d1aAccY','d1aAccZ','d1aOmegaX','i1aAccY','i1aAccZ','i1aOmegaX',inverse_kinematics_outputs];
foot_names= {'F1','F2'};

dbstop if error
close all
if exist('processed_data.csv')==0
    [data_fin] = process_data(pwd);
else
    load('processed_data.csv')
    data_fin = processed_data;
end
new_time = data_fin(:,1);
fin_mat3 = data_fin;
z = find(fin_mat3(:,8)==1);
z2 = find(fin_mat3(:,end-1)==1);

y = z(2:end)-z(1:end-1);
IQR = prctile(y,75)-prctile(y,25);
strides_to_delete = [find(y>prctile(y,75)+IQR*1.5) find(y<prctile(y,25)-IQR*1.5)];
% z = z(y<prctile(y,75)+ IQR*1.5 & y>prctile(y,25)- IQR*1.5);
% y = y(y<prctile(y,75)+ IQR*1.5 & y>prctile(y,25)- IQR*1.5);

y2 = z2(2:end)-z2(1:end-1);
strides_to_delete2 = [find(y2>prctile(y2,75)+IQR*1.5) find(y2<prctile(y2,25)-IQR*1.5)];
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
%         sample_strides = setdiff(sample_strides, strides_to_delete);
        test_strides = find(cvIndices == cv);
%         test_strides = setdiff(test_strides, strides_to_delete);
        
        % split training testing
        data_train = [];
        cycle_time = [0];
 
        HStrain = [find(data_train(:,ft*7)==1); size(data_train,1)+1];
        trainCycles = HStrain(2:end)-HStrain(1:end-1);

        for i = 1:length(sample_strides)
            ii = sample_strides(i);
            data_train = [data_train; data_in(stMark1(ii):stMark1(ii+1)-1,:)];
            stridesST.(['stride' num2str(i)]) = data_in(stMark1(ii):stMark1(ii+1)-1,[1:6,8:13]);
            
%             if TWO_FEET
%                 ft1 = [ft1 create_feature_mat(data_in(stMark1(jj):stMark1(jj+1)-1,8:16), cycle_time(end))]
%             end
            feature_mat = [feature_mat; ...
                create_feature_mat(data_in(stMark1(jj):stMark1(jj+1)-1,:), cycle_time(end))];
            cycle_time = [cycle_time size(data_in(stMark1(ii):stMark1(ii+1)-1,:),2)*SAMPLE_RATE_HZ];
        end
        feature_mat(1,end) = mean(cycle_time);
%         feature_mat = feature_mat(2:end,:);
        
        HStrain = [find(data_train(:,ft*7)==1); size(data_train,1)+1];
        HStrain2 = [find(data_train(:,setdiff([7 14],ft*7))==1); size(data_train,1)+1];
        trainCycles = HStrain(2:end)-HStrain(1:end-1);
        trainCycles2 = HStrain2(2:end)-HStrain2(1:end-1);
        trainCycles = [mean(trainCycles); trainCycles];
        trainCycles2 = [mean(trainCycles2); mean(trainCycles2); trainCycles2];
        for str = 1:length(fields(stridesST))
            ft_mat = create_feature_mat(stridesST.(['stride' num2str(i)])(:,((ft-1)*6+1):((ft-1)*6+6)), trainCycles(str));
            if TWO_FEET
                ft2 = setdiff([1,2],ft);
                ft_mat = [ft_mat create_feature_mat(stridesST.(['stride' num2str(i)])(:,((ft2-1)*6+1):((ft2-1)*6+6)), trainCycles2(str))];
            end
        end
        b = regress(trainCycles(2:end),feature_mat);
                
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
%         LRlabels_ts = [];
%         for tc = 1:length(testCycles)
%             LRlabels_ts = [LRlabels_ts linspace(0,testCycles(tc)*SAMPLE_RATE_HZ...
%                 ,testCycles(tc))./(testCycles(tc)*SAMPLE_RATE_HZ)];
%         end

        
        % Testing
        guess_vec = [];
        data_pred = [];
        mse_loss = 0;
%         if cv == 1
%             subplot(2,2,iter)
%             iter = iter+1;
%         plot([0 0],[1, 1]);
%         hold on;
%         xlabel('Actual cycle time')
%         ylabel('Predicted cycle time')
%         axis([0 1 0 1])
%         end
%         pl = 1;
        for pt_idx = 1:size(data_test,1)
%             if LRlabels_ts(pt_idx)== 0
%                 col = rand(1,3);
%                 pl = pl+1;
%             end

            if TWO_FEET
                pt_in = data_test(pt_idx,[1:6,8:13]);
            else
                pt_in = data_test(pt_idx,[1:6]);
            end
            pred_time = pt_in*b;
            mse_loss = mse_loss + (LRlabels_ts(pt_idx) - pred_time).^2;
%             if cv == 1
%                 p(pl) = scatter(LRlabels_ts(pt_idx), pred_time, 10, col);
%             end
            
        end
%         legend(p)
%         tvec = {'One Foot Training','Two Feet Training'};
%         title(['Fold ' num2str(cv), ', Foot ' num2str(ft) ', ' tvec{tf}])
        mse_loss_t = mse_loss / size(data_test,1);
        
        % sample_strides = randsample(length(stPF),round(.8*length(stPF)));
        % test_strides = setdiff(1:length(stPF), sample_strides);
        
        % labels_tr = labels(sample_strides);
        % labels_tst = labels(test_strides);
        
%         data_train=[];
%         labels_tr=[];
%         for i = 1:length(sample_strides)
%             ii = sample_strides(i);
%             %     if ii ==length(stPF)
%             if i<length(stPF)
%                 %         data_train = [data_train; data_in(stPF(ii):end,:)];
%                 %     else
%                 data_train = [data_train; data_in(stPF(i):stPF(i+1)-1,:)];
%                 labels_tr = [labels_tr; labels(stPF(i):stPF(i+1)-1)];
%             end
%             if ii ==1
%                 data_train = [data_in(1:stPF(1)-1,:); data_train];
%                 labels_tr = [labels(1:stPF(1)-1); labels_tr];
%             elseif ii==sample_strides(end)
%                 data_train = [data_train; data_in(stPF(end),:)];
%                 labels_tr = [labels_tr; labels(stPF(end):end)];
%             end
%         end
%         data_test=[];
%         labels_tst=[];
%         for k = 1:length(test_strides)-1
%             kk = test_strides(k);
%             %     if kk == length(stPF)
%             %         data_test = [data_test; data_in(stPF(kk):end,:)];
%             %     else
%             if k<length(stPF)
%                 data_test = [data_test; data_in(stPF(k):stPF(k+1)-1,:)];
%                 labels_tst = [labels_tst; labels(stPF(k):stPF(k+1)-1)];
%             end
%             if kk ==1
%                 data_test = [data_in(1:stPF(1)-1,:); data_test];
%                 labels_tst = [labels(1:stPF(1)-1); labels_tst];
%             elseif kk==test_strides(end)
%                 data_test = [data_test; data_in(stPF(end),:)];
%                 labels_tst = [labels_tst; labels(stPF(end):end)];
%             end
%             
%         end
%         %     figure;
%         %     subplot(2,1,1)
%         %     plot(1:size(data_train,1), data_train)
%         %     title("Training Data, Testing Data")
%         %     subplot(2,1,2)
%         %     plot(1:size(data_test,1), data_test)
%         gp_mat = {'m','c','y'};
% %         for k = 1:two_feet+1
%         FootSt = romanFunction(data_train);
%         [randomized_feature_matrix, labels_in] = createFeatureMatrix(FootSt, labels_tr, ft, prediction_signals, TWO_FEET, foot_names);
% 
%         Mdl.(['Foot' num2str(ft)]) = fitcdiscr(randomized_feature_matrix,labels_in);
%         %     W(:,:,k) = LDA(randomized_feature_matrix, labels_in);
% %         end
% 
%         FootSt2 = romanFunction(data_test);
% 
%         [randomized_feature_matrix_test, labels_in_test] = createFeatureMatrix(FootSt2, labels_tst, ft, prediction_signals, TWO_FEET, foot_names);
% 
%         guess_train = predict(Mdl.(['Foot' num2str(ft)]), randomized_feature_matrix);
%         guesses = predict(Mdl.(['Foot' num2str(ft)]), randomized_feature_matrix_test);
%         acc = sum(guesses == tslab)/length(guesses);
%         C = confusion.getMatrix([labels_in_test;1; 2; 3], [guesses; 3; 1; 2]);
%         
% %         [acc, C] = crossval_fun2(data_train,labels_tr, data_test, labels_tst, PLOT, TWO_FEET, new_time);
%         disp(['Fold ' num2str(cv)])
%         disp(C)
%         accv(cv) = acc;
%         %     accvec.(['Foot' num2str(ft)]).(['Fold' num2str(cv)]) = acc;
%         
%         Cstruct.(['Foot' num2str(ft)]).(['Fold' num2str(cv)]) = C;
%     accv(cv) = acc;
    end
%     Cstruct.(['Foot' num2str(ft)]).(['Fold' num2str(cv)]) = C;
%     accvec.(['Foot' num2str(ft)]) = accv;

end
% subplot(1,2,tf)
% tvec = {'One Foot Training','Two Feet Training'};
% 
% plot(accvec.Foot1)
% hold on
% plot(accvec.Foot2)
% title(tvec{tf})
% axis([-inf inf 0 1])
% xlabel('Cross Val Folds')
% ylabel('Testing Accuracy')
% legend('Foot L', 'Foot R')
end

function [ft_mat] = create_feature_mat(stride, prev_cycle_time)
% means = mean(stride);
% maxs = max(stride);
HS = sqrt(sum(stride(1,1:3).^2));

ft_mat = [prev_cycle_time HS];
end

