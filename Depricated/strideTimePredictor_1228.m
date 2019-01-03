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
USE_PREV_TIME = 1;
PREDICTION_WINDOW_START = .6;
PREDICTION_WINDOW_END = 1;

inverse_kinematics_outputs = {'r0','r1','r2','r3','gAccY','gAccZ','gVelY','gVelZ','gPosY','gPosZ'};
prediction_signals = ['aAccY','aAccZ','aOmegaX','d1aAccY','d1aAccZ','d1aOmegaX','i1aAccY','i1aAccZ','i1aOmegaX',inverse_kinematics_outputs];
foot_names= {'F1','F2'};

dbstop if error
close all
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

FootStrAll = romanFunction(data_in);
% for iters = 1:10
%     TWO_FEET = mod(
STRIDE_MARKER = 1;
figure;
iter = 1;
for tf = 1:2
    TWO_FEET = tf-1;
    accvec = [];
for ft = 1:2 % foot 1 or 2
%     stMark1 = z;
%     stMark2 = z2;
    stMark1 = find(data_in(:,7)==STRIDE_MARKER);
    stMark2 = find(data_in(:,14)==STRIDE_MARKER);
    strides_to_delete = strides_to_delete1;
    labels_full = labels1o;
    if ft==2
%         stMark1 = z2;
%         stMark2 = z;
        stMark1 = stMark2;
        stMark2 = stMark1;
        strides_to_delete = strides_to_delete2;
        labels_full = labels2o;
    end
%     stridesAll = FootStrAll.([foot_names{ft}]);
%     stridesAllOther = FootStrAll.([foot_names{setdiff(1:2,ft)}]);
    
    select_labels = labels_full(stMark1);
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
        prev_cycle_time = [mean(trainCycles); trainCycles];
%         prev_cycle_time = prev_cycle_time./max(prev_cycle_time);
        k = [];
        for tc = 1:length(trainCycles)
            LRlabels_tr = [LRlabels_tr linspace(0,trainCycles(tc)*SAMPLE_RATE_HZ...
                ,trainCycles(tc))./(trainCycles(tc)*SAMPLE_RATE_HZ)];
            k = [k; repmat(prev_cycle_time(tc),[trainCycles(tc),1])];
        end
        
        if TWO_FEET
            feature_mat = data_train(:,[1:6,8:13]);
        else
            feature_mat = data_train(:,1:6);
        end
%         for feat = 1:size(data_train,2)
%             figure
%             autocorr(data_train(:,feat))
%             title(['Feat ' num2str(feat)])
%         end
       
%         R = corrcoef(feature_mat);
%         V=diag(inv(R))';
        
%         b = ridge(LRlabels_tr',feature_mat);
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
        if ~USE_PREV_TIME
            k = zeros(size(k));
        end
        
        window_keep = sort([intersect(find(LRlabels_tr>=PREDICTION_WINDOW_START),find(LRlabels_tr<=PREDICTION_WINDOW_END))]);
        b = regress(LRlabels_tr(window_keep)',feature_mat(window_keep,:) + k(window_keep));
        
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
        if ft == 1 && cv==1
%             subplot(2,2,iter)
            
            iter = iter+1;
        end
        pl = 0;
        pred_vec = [];
        labpt = [];
        stPts = [1];
        prev_ct_vec=[];
        pred_iter = 1;
        pi_vec = [];
        for pt_idx = 1:size(data_test,1)
            if LRlabels_ts(pt_idx)== 0
                pl = pl+1;
                if pt_idx > 1
                    stPts = [stPts; pt_idx];
                    prev_ct = length(LRlabels_ts(stPts(pl-1):stPts(pl)));
                else
                    prev_ct = mean(trainCycles);
                end
            end
            prev_ct_vec(pt_idx) = prev_ct;
            if TWO_FEET
                pt_in = data_test(pt_idx,[1:6,8:13]);
            else
                pt_in = data_test(pt_idx,[1:6]);
            end
            % pt_in = 1x6; b = 6x1734
            data_pred = [data_pred; pt_in];
            if ~USE_PREV_TIME
                prev_ct_vec = zeros(size(prev_ct_vec));
            end
            if LRlabels_ts(pt_idx)>=PREDICTION_WINDOW_START
                pred_time = (data_pred(end,:) + prev_ct_vec(end)')*b;
                mse_loss = mse_loss + (LRlabels_ts(pt_idx) - pred_time(end)).^2;
                pred_vec(pred_iter) = pred_time(end);
                labpt(pred_iter) = select_labels(test_strides(pl));
                pred_iter = pred_iter +1;
                pi_vec(pred_iter) = pt_idx;
            end
            
            %             if cv==1 && ft ==1
            %                 scatter(LRlabels_ts(pt_idx), pred_time, 10, col,'Filled');
            %             end
        end
        %         legend(p)
        gps = {'Upstairs','Downstairs','Level Ground'};
        pic_title = ['Possible Paper Figures/' deblank(strrep([date '_linreg' char('_offset'*USE_PREV_TIME) '_' char('window'*(PREDICTION_WINDOW_START~=0))],'-','_'))];
        if ft ==1
            figure(tf)
            figure(15)
            plot([0 1],[0 1],'k','LineWidth',2);
            hold on;
            xlabel('Actual cycle time')
            ylabel('Predicted cycle time')
            axis([0 1 -inf inf])
            labpt_names = gps(labpt);
            gscatter(LRlabels_ts(LRlabels_ts>=PREDICTION_WINDOW_START), pred_vec, labpt_names');
            tvec = {'One Foot Training','Two Feet Training'};
            title([tvec{tf}])
            if cv==1 || cv==5
                saveas(gcf, [pic_title '_' num2str(tf) 'ft_CV' num2str(cv) '.png'])
            end

        end
        mse_loss_t(cv) = mse_loss / (pred_iter-1);
        
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
linkaxes([ax(1),ax(2)],'y')
saveas(gcf,[pic_title '_mse.png'])
