%% Notes from oct 26 meeting
clear all

% Define parameters
POST_SWING_CUTOFF_TIME_S=0.2;
SAMPLE_RATE_HZ = 100;
POST_SWING_CUTOFF_SAMPLES = int32(POST_SWING_CUTOFF_TIME_S * SAMPLE_RATE_HZ);

TWO_FEET = 0;
FOLDS = 5;
PLOT = false;
USE_PREV_TIME = 1;
PREDICTION_WINDOW_START = .6;
PREDICTION_WINDOW_END = 1;
NUM_TIMEPOINTS_VEC = 0; % set to 1 to not use averaging; make vector to check multiple timepoints
CNN = 0;
PRED_TIME = 0;
PRED_TERRAIN = 1;
LDA = 1;
REGRESSION = 0;

raw_sensor_outputs = {'a1RawX','a2RawY','a3RawZ','g1RawX','g2RawY','g3RawZ'};
inverse_kinematics_outputs = {'r0','r1','r2','r3','gAccY','gAccZ','gVelY','gVelZ','gPosY','gPosZ'};
derivative_outputs = {'d1aAccX','d1aAccY','d1aAccZ','d1aOmegaX','d1aOmegaY','d1aOmegaZ'};
integral_outputs = {'i1aAccX','i1aAccY','i1aAccZ','i1aOmegaX','i1aOmegaY','i1aOmegaZ'};
prediction_signals = {'aAccY','aAccZ','aOmegaX','d1aAccY','d1aAccZ','d1aOmegaX','i1aAccY','i1aAccZ','i1aOmegaX'};
foot_names= {'F1','F2'};

prediction_signals = raw_sensor_outputs;
dbstop if error
close all
dt = '22-Dec-2018';
dbstop if error
close all
% load data
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

% fix discrepancy between feet

y = z(2:end)-z(1:end-1);
IQR = prctile(y,75)-prctile(y,25);

y2 = z2(2:end)-z2(1:end-1);

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


STRIDE_MARKER = 1;
figure;
iter = 1;
for tf = 1:2
    TWO_FEET = tf-1;
    accvec = [];
    mse_tot_nt = [];
    for nt = 1:length(NUM_TIMEPOINTS_VEC)
        NUM_TIMEPOINTS = NUM_TIMEPOINTS_VEC(nt);
        for ft = 1:2
            stMark1 = find(data_in(:,7)==STRIDE_MARKER);
            stMark2 = find(data_in(:,14)==STRIDE_MARKER);
            strides_to_delete = strides_to_delete1;
            labels_full = labels1o;
            if ft==2
                stMark1 = stMark2;
                stMark2 = stMark1;
                strides_to_delete = strides_to_delete2;
                labels_full = labels2o;
            end
            select_labels = labels_full(stMark1);
            last_lab = select_labels(end);
            select_labels = select_labels(2:end); %***have labels as next startPF
            cvIndices = crossvalind('Kfold',length(stMark1)-1,FOLDS, 'Min',3);
            
            for cv = 1:FOLDS
                sample_strides = find(any(cvIndices == setdiff(1:FOLDS,cv),2));
                sample_strides = setdiff(sample_strides, strides_to_delete);
                test_strides = find(cvIndices == cv);
                test_strides = setdiff(test_strides, strides_to_delete);
                
                trlab = select_labels(sample_strides);
                tslab = select_labels(test_strides);
                
                %% Training
                data_train = [];
                terrain_labels_tr=[];
                for i = 1:length(sample_strides)
                    ii = sample_strides(i);
                    data_train = [data_train; data_in(stMark1(ii):stMark1(ii+1)-1,:)];
                    terrain_labels_tr = [terrain_labels_tr; repmat(trlab(i),[length(stMark1(ii):stMark1(ii+1)-1),1])];
                end
                
                HStrain = [find(data_train(:,ft*7)==1); size(data_train,1)+1];
                trainCycles = HStrain(2:end)-HStrain(1:end-1);
                LRlabels_tr = [];
                prev_cycle_time = [mean(trainCycles); trainCycles];
                k = [];
                for tc = 1:length(trainCycles)
                    LRlabels_tr = [LRlabels_tr linspace(0,trainCycles(tc)*SAMPLE_RATE_HZ...
                        ,trainCycles(tc))./(trainCycles(tc)*SAMPLE_RATE_HZ)];
                    k = [k; repmat(prev_cycle_time(tc),[trainCycles(tc),1])];
                end
                
                if ft == 2
                    feature_mat = data_train(:,[8:14,1:7]);
                end
                if TWO_FEET
                    feature_mat = data_train(:,[1:6,8:13]);
                else
                    feature_mat = data_train(:,1:6);
                end

                [scaled_sensor_mat,integral_mat,derivative_mat] = scale_batch(feature_mat,...
                    [find(data_train(:,7)==1); size(data_train,1)],...
                    find(data_train(:,14)==1));
                
                feature_mat = calculate_prediction_signals(integral_mat, ...
                    derivative_mat, scaled_sensor_mat, feature_mat, prediction_signals);
                
                if NUM_TIMEPOINTS>1 || NUM_TIMEPOINTS == 0
%                     if NUM_TIMEPOINTS == 0
%                         stride_idxs = HStrain;
                    feat_means=[];
                    labels_tr_means=[];
                    terrain_lab_means_tr = [];
                    k_means = [];
                    for fm = 1:size(feature_mat,1)-NUM_TIMEPOINTS-1
                        if fm > length(HStrain)-1 && NUM_TIMEPOINTS == 0
                             break
                        end
                        if NUM_TIMEPOINTS == 0
%                             if fm>6
%                           %% might need to change this for different window      
                            idx1 = round(HStrain(fm) + PREDICTION_WINDOW_START*(HStrain(fm+1)-HStrain(fm))); 
                            idx2 = round(HStrain(fm+1) - (1-PREDICTION_WINDOW_END)*(HStrain(fm+1)-HStrain(fm)));
                            if fm == length(HStrain)-1
                                idx2 = size(feature_mat,1);
                            end
                        else
                            idx1 = fm; idx2 = fm+NUM_TIMEPOINTS-1;
                        end
                        if CNN
                           feat_means(:,:,:,fm) = feature_mat(idx1:idx2,:);
                        else
                            feat_means(fm,:) = [mean(feature_mat(idx1:idx2,:)) ...
                                max(feature_mat(idx1:idx2,:)) range(feature_mat(idx1:idx2,:))];
                        end
                        labels_tr_means(fm) = median(LRlabels_tr(idx1:idx2));
                        k_means(fm) = median(k(idx1:idx2));
                        terrain_lab_means_tr(fm) = median(terrain_labels_tr(idx1:idx2));
                    end
                    k = k_means';
                    feature_mat = feat_means;
                    LRlabels_tr = labels_tr_means;
                    terrain_labels = terrain_lab_means_tr;
                end
                
                if ~USE_PREV_TIME
                    k = zeros(size(k));
                end
                if NUM_TIMEPOINTS ~=0
                    window_keep = sort([intersect(find(LRlabels_tr>=PREDICTION_WINDOW_START),find(LRlabels_tr<=PREDICTION_WINDOW_END))]);
                    LRlabels_tr = LRlabels_tr(window_keep); terrain_labels = terrain_labels(window_keep);
                    k = k(window_keep);
                end
                
                if PRED_TERRAIN
                    LABELS = terrain_labels';
                elseif PRED_TIME
                    LABELS = LRlabels_tr';
                elseif PRED_TERRAIN && PRED_TIME
                    if CNN
                        terrain_labels = make_one_hot(round(terrain_labels)',[1,2,3]);
                    end
                    LABELS = [LRlables_tr' terrain_labels];
                end
                    
                if CNN 
                    
                    layers = [ ...
                        imageInputLayer([NUM_TIMEPOINTS length(prediction_signals) 1]) % 4x6x1 --> 3x4x3 = 36
                        convolution2dLayer([2 3],3)
                        reluLayer
                        fullyConnectedLayer(4)
                        regressionLayer];
%                     terrain_labels_tr_OH = make_one_hot(round(terrain_labels_tr)',[1,2,3]);
                    options = trainingOptions('rmsprop', ...
                        'MaxEpochs',10, ...
                        'Shuffle','every-epoch', ...
                        'ValidationFrequency',30, ...
                        'Verbose',false);
%                         'ValidationData',{ftmat_test, [labels_ts_fin' labtest22_OH]});
%                     LABELS = [make_one_hot(round(LABELS(:,1))',[1,2,3]) LABELS(:,end)'];
%                     net = trainNetwork(feature_mat(:,:,:,window_keep),[LRlabels_tr(window_keep)' terrain_labels_tr_OH(window_keep,:)],layers,options);
                    
                    net = trainNetwork(feature_mat,LABELS,layers,options);
                elseif REGRESSION
%                     b = regress(LRlabels_tr(window_keep)',feature_mat(window_keep,:) + k(window_keep));
                    
                    b = regress(LABELS,feature_mat + k);
                elseif LDA
                    Mdl = fitcdiscr(feature_mat + k, LABELS);
                end

                %% Testing
                data_test = [];
                labtest2 = [];
                terrain_labels_ts = [];
                for j = 1:length(test_strides)
                    jj = test_strides(j);
                    data_test = [data_test; data_in(stMark1(jj):stMark1(jj+1)-1,:)];
                    
                    terrain_labels_ts = [terrain_labels_ts; repmat(tslab(j),...
                        [length(stMark1(jj):stMark1(jj+1)-1),1])];

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
                tstlab = [];
                terrlab = [];
                prev_int = zeros([1,length(prediction_signals)]); prev_deriv = zeros([1,length(prediction_signals)]);
                int_sig = []; deriv_sig = []; sc_sig = [];
                if NUM_TIMEPOINTS ~= 0
                    num_timepoints = NUM_TIMEPOINTS;
                else
                    num_timepoints = inf;
                end
                
                for pt_idx = 1:size(data_test,1)
                    if LRlabels_ts(pt_idx)== 0
                        
                        pl = pl+1;
                        if pt_idx > 1
                            if size(prev_int)>0
                                int_offset = mean(prev_int(stPts(end-1:end),:));
                                prev_int(stPts(end-1:end),:) = prev_int(stPts(end-1:end),:) - int_offset;
                            end
                            stPts = [stPts; pt_idx];
%                             int_offset = mean(prev_int(stPts(end-1:end),:));
%                             prev_int(stPts(end-1:end),:) = prev_int(stPts(end-1:end),:)...
%                                 - mean(prev_int(stPts(end-1:end),:));
                            prev_ct = length(LRlabels_ts(stPts(pl-1):stPts(pl)));
                            if NUM_TIMEPOINTS == 0
                                num_timepoints = stPts(end) - stPts(end-1);
                                
                            end
                        else
                            prev_ct = mean(trainCycles);
                        end
                    end
                    prev_ct_vec(pt_idx) = prev_ct;
                    if ft ==2
                        pt_in = data_test(pt_idx,[8:13,1:6]);
                    else
                        pt_in = data_test(pt_idx,[1:6,8:13]);
                    end
                    if ~TWO_FEET
                        pt_in = pt_in(:,1:6);
                    end
                    % pt_in = 1x6; b = 6x1734
                    %             data_pred = [data_pred; pt_in];
                    if ~USE_PREV_TIME
                        prev_ct_vec = zeros(size(prev_ct_vec));
                    end
                    if LRlabels_ts(pt_idx)>=PREDICTION_WINDOW_START || (LRlabels_ts(pt_idx)==0 && NUM_TIMEPOINTS == 0)
                        data_pred = [data_pred; pt_in];
                        if size(data_pred,1)>=num_timepoints
%                             if pt_idx == num_timepoints
%                                 prev_int = zeros(size(data_pred)); prev_deriv = zeros(size(data_pred));
%                             end
                            for feat = 1:length(pt_in)
                                 sc_sig(:,feat) = scaled(data_pred(:,feat),feat);
                                 int_sig(:,feat) = integrate(sc_sig(:,feat), prev_int(end,feat));
                                 deriv_sig(:,feat) = derivative(sc_sig(:,feat), prev_deriv(end,feat));
                            end
                            prev_int = [prev_int; int_sig]; prev_deriv = [prev_deriv; deriv_sig];
                            data_pred_fin = calculate_prediction_signals(int_sig, deriv_sig, sc_sig,  data_pred, prediction_signals);
                            if num_timepoints>1
                                % start here!
                                % add dsig input into dsig function and
                                % same w/ integral

%                                 pt_pred = calculate_prediction_signals(data_pred, prediction_signals, prev_pt);
                                if CNN
                                    pt_pred = reshape(data_pred_fin, [size(data_pred_fin),1,1]);
                                else 
                                    pt_pred = [mean(data_pred_fin) max(data_pred_fin) range(data_pred_fin)];
                                end
                                prev_ct1 = median(prev_ct_vec((end-num_timepoints+1):end));
                                tstlab = [tstlab median(LRlabels_ts((pt_idx-num_timepoints+1):pt_idx))];
                                terrlab = [terrlab median(terrain_labels_ts((pt_idx-num_timepoints+1):pt_idx))];
                            else
                                pt_pred = pt_in;
                                prev_ct1 = prev_ct_vec(end);
                                tstlab = [tstlab LRlabels_ts(pt_idx)];
                                terrlab = [terrlab terrain_labels_ts(pt_idx)];
                            end
                            if PRED_TERRAIN && PRED_TIME && CNN
                                guess = predict(net, pt_pred);
                                pred_time = guess(:,1);
                                label_guess = guess(:,2:end);
                                [probs, pred_terr] = max(exp(label_guess)./sum(exp(label_guess),2),[],2);
                            elseif REGRESSION
                                pred_time = (pt_pred + prev_ct1')*b;
                            elseif LDA
                                pred_terr = predict(Mdl, pt_pred + prev_ct1');
                            elseif CNN && ~PRED_BOTH && PRED_TIME
                                pred_time = predict(net, pt_pred);
                            elseif CNN && ~PRED_BOTH && PRED_TERRAIN
                                pred_terr = predict(net, pt_pred + prev_ct1');
                            end
                            
%                             pred_ti me = (pt_pred + prev_ct1')*b;
                            if PRED_TIME
                                mse_loss = mse_loss + (tstlab(end) - pred_time).^2;
                                pred_vec(pred_iter) = pred_time;
                                labpt(pred_iter) = select_labels(test_strides(pl));
                            elseif PRED_TERRAIN
                                pred_vec(pred_iter) = pred_terr;
%                                 acc = sum(pred_terr == terrlab)/length(pred_terr);
%                                 C = confusion.getMatrix([terrlab;1; 2; 3], [pred_terr; 3; 1; 2],0);
                            end
%                             pred_vec(pred_iter) = pred_time;
%                             labpt(pred_iter) = select_labels(test_strides(pl));
                            pred_iter = pred_iter +1;
                            %                     pi_vec(pred_iter) = pt_idx;
                            data_pred = data_pred(2:end,:);
                            if NUM_TIMEPOINTS == 0
                                data_pred = [pt_in];
                            end
                        end
                    end
                end
                if PRED_TERRAIN
                    acc(cv) = sum(pred_vec==terrlab)/length(pred_vec);
                end
                %         legend(p)
                if PRED_TIME
                gps = {'Upstairs','Downstairs','Level Ground'};
                pic_title = ['Possible Paper Figures/' deblank(strrep([date '_linreg' ...
                    char('_offset'*USE_PREV_TIME) '_' char('window'*(PREDICTION_WINDOW_START~=0)) ...
                    char('arima'*ARIMA) '_' num2str(NUM_TIMEPOINTS)],'-','_'))];
                if ft ==1
                    figure(tf*nt)
%                     figure(15)
                    plot([0 1],[0 1],'k','LineWidth',2);
                    hold on;
                    xlabel('Actual cycle time')
                    ylabel('Predicted cycle time')
                    axis([0 1 -inf inf])
                    labpt_names = gps(labpt);
                    gscatter(tstlab, pred_vec, labpt_names');
                    tvec = {'One Foot Training','Two Feet Training'};
                    title([tvec{tf} ', ' num2str(NUM_TIMEPOINTS) ' Timepoints'])
                    if cv==1 || cv==5
                        saveas(gcf, [pic_title '_' num2str(tf) 'ft_CV' num2str(cv) '.png'])
                    end
                end
                acc(cv) = mse_loss / (pred_iter-1);    
                end
%                 mse_loss_t(cv) = mse_loss / (pred_iter-1);
                
            end
            mse_tot(ft,:) = acc;
        end
        mse_tot_nt(:,:,nt) = mse_tot;
    end
    figure(2*nt+1);
    ax(tf) = subplot(1,2,tf);
    tvec = {'One Foot Training','Two Feet Training'};
    ft_1 = squeeze(mse_tot_nt(1,:,:)); ft_2 = squeeze(mse_tot_nt(2,:,:));
    title(tvec{tf})
    if size(mse_tot_nt,3)>1
%         [X,Y] = meshgrid(1:size(mse_tot_nt,2),1:size(mse_tot_nt,3));
%         surf(X,Y,ft_1')
        plot(mean(ft_1),'*k')
        hold on 
        boxplot(ft_1)
        plot(mean(ft_2),'*r')
        boxplot(ft_2)
%         surf(X,Y,ft_2')
        yticks(1:length(NUM_TIMEPOINTS_VEC))
        yticklabels(NUM_TIMEPOINTS_VEC)
        xlabel('Number of Timpoints')
        ylabel('MSE Loss')
        title([tvec{tf} ', Means over CV Folds'])
%         xlabel('Cross Val Folds')
    else

        plot(mse_tot(1,:))
        hold on
        plot(mse_tot(2,:))
        xlabel('Cross Val Folds')
        ylabel('MSE Loss')
    end
%     title(tvec{tf})
    %     axis([-inf inf -inf inf])
    legend('Foot L', 'Foot R')
%     xticklabels(1:FOLDS)
end
linkaxes([ax(1),ax(2)],'y')
saveas(gcf,[pic_title '_mse.png'])

function [dsig] = derivative(signal, dsig)
FILTB = 0.05;
FILTA = 0.95;

% dsig = 0;
for j = 2:length(signal)
    dsig(j) = FILTB*(signal(j) - signal(j-1)) + FILTA*dsig(j-1);
end
end

function [ssig] = scaled(signal, idx)
GRAVITY_MPS2 = 9.8;
RAD_PER_DEG = pi/180;
GYRO_LSB_PER_DPS = 32.8; %per http://dephy.com/wiki/flexsea/doku.php?id=units
ACCEL_LSB_PER_G = 8192; %per http://dephy.com/wiki/flexsea/doku.php?id=units
ACCEL_LSB_PER_MPS2 = ACCEL_LSB_PER_G / GRAVITY_MPS2;
GYRO_LSB_PER_RAD = GYRO_LSB_PER_DPS / RAD_PER_DEG;

scale_factors = {ACCEL_LSB_PER_MPS2, -1.0*ACCEL_LSB_PER_MPS2, ACCEL_LSB_PER_MPS2, ...
    GYRO_LSB_PER_RAD, GYRO_LSB_PER_RAD, GYRO_LSB_PER_RAD};
ssig = signal / scale_factors{idx};
end

function [isig] = integrate(signal, isig)
% isig = 0;
for j = 2:length(signal)
    isig(j) = isig(j-1) + signal(j);
end
end

function [scaled_sensor_mat,integral_mat,derivative_mat] = scale_batch(mat, varargin)
DEFAULT_ZVUP_TIME_S = 0.05;
SAMPLE_RATE_HZ = 100;
DEFAULT_ZVUP_SAMPLES = DEFAULT_ZVUP_TIME_S * SAMPLE_RATE_HZ;

UPPER_ACCNORM_THRESH_SQ = 102.01;
LOWER_ACCNORM_THRESH_SQ = 90.25;
for feat = 1:size(mat,2)
    fmod = mod(feat,6);
    if fmod==0
        fmod = 6;
    end
    scaled_sensor_mat(:,feat) = scaled(mat(:,feat), fmod);
    integral_mat(:,feat) = integrate(scaled_sensor_mat(:,feat),0);
    derivative_mat(:,feat) = derivative(scaled_sensor_mat(:,feat),0);
end

if ~isempty(varargin)
    initStance1 = varargin{1};
    initSwing1 = initStance1(1:end-1) + .6*(initStance1(2:end) - initStance1(1:end-1));
    
    initStance2 = varargin{2};
    initSwing2 = initStance2(1:end-1) + .6*(initStance2(2:end) - initStance2(1:end-1));
end
for h = 1:(size(mat,2)/6)
    initStance = varargin{h};
    initSwing = initStance(1:end-1) + .6*(initStance(2:end) - initStance(1:end-1));
    if h == 2
        initSwing = [initSwing; size(mat,1)];
    end
    for k = 1:length(initSwing)
        stanceSamples = initStance:initSwing;
        accNormSq = zeros(length(stanceSamples),1);
        foundFirstFootStatic = 0;
        latestFootStaticSample = 2;
        resetTrigger = 0;
        for j=stanceSamples(2:end)
            accNormSq(j) = sumsqr([scaled_sensor_mat(j,(h-1)*6+2) scaled_sensor_mat(j,(h-1)*6+3)]);
            if (j - latestFootStaticSample > DEFAULT_ZVUP_SAMPLES && ~foundFirstFootStatic)
                latestFootStaticSample = j;
                resetTrigger = 1;
            end
            % optimal case
            if (accNormSq(j) < UPPER_ACCNORM_THRESH_SQ && ...
                    accNormSq(j) > LOWER_ACCNORM_THRESH_SQ)
                foundFirstFootStatic = 1;
                latestFootStaticSample = j;
                resetTrigger = 2;
            end
        end
        % reset integral outputs
        if h == 2 && k == length(initSwing)
%             integral_mat(initStance(k):end,:) = integral_mat(initStance(k):end,:) - integral_mat(latestFootStaticSample,:);
            integral_mat(initStance(k):end,:) = integral_mat(initStance(k):end,:) - mean(integral_mat(initStance(k):end,:));
        else
%             integral_mat(initStance(k):initStance(k+1),:) = integral_mat(initStance(k):initStance(k+1),:) - integral_mat(latestFootStaticSample,:);
            integral_mat(initStance(k):initStance(k+1),:) = integral_mat(initStance(k):initStance(k+1),:) - mean(integral_mat(initStance(k):initStance(k+1),:));        
        end
    end
end

end
function [final_mat] = calculate_prediction_signals(int_sig, deriv_sig ,scaled_sig, orig_sig, prediction_signals)
for feat = 1:length(prediction_signals)
    fmod = mod(feat,6);
    if fmod==0
        fmod = 6;
    end
    xyz = [prediction_signals{feat}(end)==('X'), prediction_signals{feat}(end)==('Y'), ...
        prediction_signals{feat}(end)==('Z')];
    ftt = ~contains(prediction_signals{feat}, 'Acc')*3+(1*find(xyz));
    if prediction_signals{feat}(1)=='d'
        final_mat(:,feat) = deriv_sig(:,ftt);
    elseif prediction_signals{feat}(1)=='i'
        final_mat(:,feat) = int_sig(:,ftt);
    elseif prediction_signals{feat}(1)=='a'
        final_mat(:,feat) = scaled_sig(:,ftt);
    else
        final_mat(:,feat) = orig_sig(:,ftt);
    end
end
end
