%% Notes from oct 26 meeting

% flops: layers * timesteps * 8 * 2 * hiddenSize * minibatch * (hiddenSize + 1)
% layers = 1 * NUM_TIMEPOINTS * 8 * 2 * 
clear all

% Define parameters
POST_SWING_CUTOFF_TIME_S=0.2;
SAMPLE_RATE_HZ = 100;
POST_SWING_CUTOFF_SAMPLES = int32(POST_SWING_CUTOFF_TIME_S * SAMPLE_RATE_HZ);

FOLDS = 5;
PLOT = false;
% USE_PREV_TIME = 1;
PREDICTION_WINDOW_START = .6;
PREDICTION_WINDOW_END = 1;
NUM_TIMEPOINTS_VEC = [5 9 13]; % set to 1 to not use averaging; make vector to check multiple timepoints
CNN = 0;
LDA = 1;
LSTM=0;
plot_pts = 0;
if length(NUM_TIMEPOINTS_VEC) > 1
    CONTINUOUS = 1;
else
    CONTINUOUS = 0;
end
WINDOW_START = .6;
WINDOW_END = 1;

raw_sensor_outputs = {'a1Raw','a2Raw','a3Raw','g1Raw','g2Raw','g3Raw'};
inverse_kinematics_outputs = {'r0','r1','r2','r3','gAccY','gAccZ','gVelY','gVelZ','gPosY','gPosZ'};
derivative_outputs = {'d1aAccX','d1aAccY','d1aAccZ','d1aOmegaX','d1aOmegaY','d1aOmegaZ'};
integral_outputs = {'i1aAccX','i1aAccY','i1aAccZ','i1aOmegaX','i1aOmegaY','i1aOmegaZ'};
prediction_signals = {'aAccY','aAccZ','aOmegaX','d1aAccY','d1aAccZ','d1aOmegaX','i1aAccY','i1aAccZ','i1aOmegaX'};
if CONTINUOUS
    prediction_signals = setdiff(prediction_signals, 'HSmag');
end

foot_names= {'F1','F2'};

prediction_signals = raw_sensor_outputs;
dbstop if error
close all
dt = '22-Dec-2018';
dbstop if error
close all
% load data
if exist([dt '-processed_data 3.csv'])==0
    [data_fin] = process_data2([pwd '/Raw Data']);
else
    load([dt '-processed_data 3.csv'])
    data_fin = X22_Dec_2018_processed_data_3;
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
        for ff = 1:2
            fv = [2 1]; ft = fv(ff);
            
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
                
                % split training testing
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
                for tc = 1:length(trainCycles)
                    LRlabels_tr = [LRlabels_tr linspace(0,trainCycles(tc)*SAMPLE_RATE_HZ...
                        ,trainCycles(tc))./(trainCycles(tc)*SAMPLE_RATE_HZ)];
                end
                
                if ft == 1
                    if CONTINUOUS
                        FootSt = romanFunction_Continuous(data_train);
                    else
                        FootSt = romanFunction(data_train);
                    end
                else
                    if CONTINUOUS
                        FootSt = romanFunction_Continuous([data_train(:,8:14) data_train(:,1:7)]);
                    else
                        FootSt = romanFunction([data_train(:,8:14) data_train(:,1:7)]);
                    end
                end
                strides = FootSt.F1;
                stridesOther = FootSt.F2;
                
                HStrain = [find(data_train(:,ft*7)==1); size(data_train,1)+1];
                trainCycles = HStrain(2:end)-HStrain(1:end-1);
%                 LRlabels_tr = [];
                prev_cycle_time = [mean(trainCycles); trainCycles];

                if CONTINUOUS
                	ftmat_train = createFeatureMatrix2_Continuous(strides, ...
                        stridesOther, prediction_signals,TWO_FEET, NUM_TIMEPOINTS, CNN || LSTM);    
                    terrain_labels_tr = terrain_labels_tr(round(NUM_TIMEPOINTS/2):(end-round(NUM_TIMEPOINTS/2)));
                else
                    ftmat_train = createFeatureMatrix2(strides, stridesOther, ...
                        prediction_signals, WINDOW_START, WINDOW_END, TWO_FEET, CNN || LSTM);
                    terrain_labels_tr = trlab;
                end
                if LSTM
                    numFeatures = tf*6;
                    numHiddenUnits = 125;
                    numClasses = 3;
                    
                    options = trainingOptions('adam', ...
                        'InitialLearnRate', 1e-4,...
                        'MaxEpochs',20, ...
                        'Shuffle','every-epoch', ...
                        'ValidationFrequency',30, ...
                        'Verbose',false);
%                            'Plots','training-progress');

                    layers = [ ...
                        sequenceInputLayer(numFeatures)
                        lstmLayer(numHiddenUnits,'OutputMode','last')
                        fullyConnectedLayer(numClasses)
                        softmaxLayer
                        classificationLayer];
                    ftmat_train = cellfun(@(x) x', ftmat_train, 'uni',0)';
                     Mdl = trainNetwork(ftmat_train,categorical(terrain_labels_tr),layers,options);
                % flops: layers * timesteps * 8 * 2 * hiddenSize * minibatch * (hiddenSize + 1)
                % flops = 1 * NUM_TIMEPOINTS * 8 * 2 * numHiddenUnits *
                % length(ftmat_train) * (numHiddenUnits + 1) +
                % numHiddenUnits * numClasses * NUM_TIMEPOINTS * 

                elseif CNN
                    layers = [ ...
                        imageInputLayer([NUM_TIMEPOINTS (tf*length(prediction_signals)) 1]) % 4x6x1 --> 3x4x3 = 36
                        convolution2dLayer([2 3],3)
                        reluLayer
                        fullyConnectedLayer(1)
                        softmaxLayer
                        classificationLayer];
                    if iscell(ftmat_train) 
                        maxlen = max(cellfun('size',ftmat_train,1));
                        padded_cell = cellfun(@(x) padarray(x,maxlen-size(x,1),0,'post'), ftmat_train,'uni',0);
                        ftmat_train = cat(4, padded_cell{:});
                    end
                    rperm = randperm(size(ftmat_train,3));
                    ftmat_train = ftmat_train(:,:,:,rperm);
                    tr_labels = terrain_labels_tr(rperm);
                    options = trainingOptions('rmsprop', ...
                        'MaxEpochs',500, ...
                        'InitialLearnRate', .0001,...
                        'Shuffle','every-epoch', ...
                        'ValidationFrequency',30, ...
                        'Verbose',false);
%                     'ValidationData',{ftmat_test, categorical(ts_labels)},...
%                     'Plots','training-progress'
%                     );

                    Mdl = trainNetwork(ftmat_train,categorical(tr_labels),layers,options);
                else
                    rperm = randperm(size(ftmat_train,1));
                    ftmat_train = ftmat_train(rperm,:);
                    tr_labels = terrain_labels_tr(rperm);
                    Mdl = fitcdiscr(ftmat_train, tr_labels);
                end
%                 if ~CONTINUOUS
%                     tr_labels = trlab(rperm);
%                 end
                
                
%                 window_keep = sort([intersect(find(LRlabels_tr>=PREDICTION_WINDOW_START),find(LRlabels_tr<=PREDICTION_WINDOW_END))]);
                
                data_test = [];
                labtest2 = [];
                terrain_labels_ts = [];
                for j = 1:length(test_strides)
                    jj = test_strides(j);
                    data_test = [data_test; data_in(stMark1(jj):stMark1(jj+1)-1,:)];
                    
                    terrain_labels_ts = [terrain_labels_ts; repmat(tslab(j),...
                        [length(stMark1(jj):stMark1(jj+1)-1),1])];

                end
                if ft == 2
                    data_test = data_test(:,[8:14, 1:7]);
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
                if CONTINUOUS
                    tslab = [];
                end
%                     tslab = terrain_labels_ts(round(NUM_TIMEPOINTS/2)+1:(end-(round(NUM_TIMEPOINTS/2)-1)));
%                 end
                
                %% Testing
            guess_vec = [];
            data_pred = [];

            labiter = 1;
            start = 1;
            gp_mat = {'r','g','b'};
            rw_mat = {'r','g'};
            gp_matOther = {'m','c','y'};
            for pt_idx = 1:size(data_test,1)
                pt_in = data_test(pt_idx,:);
                data_pred = [data_pred; pt_in];
                if pt_idx == 1
                    continue
                end
                
                
                if (~CONTINUOUS && (pt_in(7)==STRIDE_MARKER || pt_idx == size(data_test,1))) || (CONTINUOUS && size(data_pred,1)>NUM_TIMEPOINTS+1)

                    if CONTINUOUS
                        FtMat = romanFunction_Continuous(data_pred(1:end-1,:));
                        stridesTest = FtMat.F1; stridesTestOther = FtMat.F2;
                        ftmat_test = createFeatureMatrix2_Continuous(stridesTest, stridesTestOther, prediction_signals,TWO_FEET,NUM_TIMEPOINTS, CNN || LSTM);
                        data_pred = data_pred(2:end,:);
                        tslab(pt_idx - (NUM_TIMEPOINTS+1),:) = round(median(terrain_labels_ts((pt_idx-NUM_TIMEPOINTS):(pt_idx-1))));
                    else
                        FtMat = romanFunction(data_pred(1:end-1,:));
                        stridesTest = FtMat.F1; stridesTestOther = FtMat.F2;
                        ftmat_test = createFeatureMatrix2(stridesTest, stridesTestOther, prediction_signals, WINDOW_START, WINDOW_END, TWO_FEET, CNN || LSTM);
                        data_pred = [pt_in];
                    end
                    if LSTM
%                         tic
                        ftmat_test = cellfun(@(x) x', ftmat_test, 'uni',0)';
                        [~,guess] = max(predict(Mdl, ftmat_test));
%                         toc
                    elseif CNN
                        ftmat_test = cell2mat(ftmat_test);
                        guess = predict(Mdl, ftmat_test);
                    else
                        guess = predict(Mdl, ftmat_test);
                    end
                    guess_vec = [guess_vec; guess];
                    if plot_pts
                        local_t = start:length(stridesTest.a1Raw)+start-1;
                        local_tOther = start:length(stridesTestOther.a1Raw)+start-1;
                        for iter =1:length(prediction_signals)
                            figure(iter);
                            hold on;
                            window_start = round(WINDOW_START*length(stridesTest.a1Raw));
                            window_end = (WINDOW_END*length(stridesTest.a1Raw))
                            plot(local_t(window_start:window_end),stridesTest.(prediction_signals{iter})(window_start:window_end),rw_mat{(tslab(labiter)==guess)+1})
%                             if TWO_FEET
%                                 window_start0 = 30;
%                                 window_end0 = window_start0 + 10;
%                                 plot(local_tOther(window_start0:window_end0),stridesTestOther.(prediction_signals{iter})(window_start0:window_end0),[gp_mat{tslab(labiter)} '--'])
%                                 legend('Foot Pred','Foot Other')
%                             end
                            title(prediction_signals{iter})
                            hold on;
                            ax = gca;
                        end
                    end
                    start = length(stridesTest.a1Raw)+start;
                    labiter = labiter+1;
                    
                end
            end
            if PLOT 
                figure(5)
                x = 0:.05:2*pi;
                y = sin(x);
                z = zeros(size(x));
                col = x;  % This is the color, vary with x in this case.
                surface([x;x],[y;y],[z;z],[col;col],...
                        'facecol','no',...
                        'edgecol','interp',...
                        'linew',2);
                corr = find(tslab==guess_vec); incorr = find(tslab~=guess_vec);
                t = 1:length(tslab);
                plt_data = data_test((round(NUM_TIMEPOINTS/2)+1:end-round(NUM_TIMEPOINTS/2)),1);
                stride_pts = find(data_test((round(NUM_TIMEPOINTS/2)+1:end-round(NUM_TIMEPOINTS/2)),7)~=0);
                z = zeros(size(plt_data'));
                col = (tslab==guess_vec)';
                surface([t;t],[plt_data';plt_data'],[z;z],[col;col],...
                    'facecol','no',...
                    'edgecol','interp',...
                    'linew',2);
                plot(t(corr),plt_data(corr),'g')
                hold on
                plot(t(incorr),plt_data(incorr),'r')
                scatter(t(stride_pts),plt_data(stride_pts),'k.')
                legend('Correct','Incorrect','Stride Marker')
%                 set(h, 'linestyle', '-');
            end
            acc = sum(guess_vec==tslab)/length(guess_vec);
            C = confusion.getMatrix([tslab;1; 2; 3], [guess_vec; 3; 1; 2],0);
            disp('Rolling predictions:')
            disp(acc)
            disp(['Fold ' num2str(cv)])
            disp(C)
            
            
            accv(cv) = acc;
                
            end
            mse_tot(ft,:) = accv;
        end
        mse_tot_nt(:,:,nt) = mse_tot;
    end
    figure(2*nt+1);
    ax(tf) = subplot(1,2,tf);
    
    tvec = {'One Foot Training','Two Feet Training'};
    ft_1 = squeeze(mse_tot_nt(1,:,:)); ft_2 = squeeze(mse_tot_nt(2,:,:));
    title(tvec{tf})
    if size(mse_tot_nt,3)>1
        axis([NUM_TIMEPOINTS_VEC(1)-1 NUM_TIMEPOINTS_VEC(end)+1 0 1])
%         [X,Y] = meshgrid(1:size(mse_tot_nt,2),1:size(mse_tot_nt,3));
%         surf(X,Y,ft_1')
        errorbar(NUM_TIMEPOINTS_VEC,mean(ft_1),std(ft_1));
        hold on 
        errorbar(NUM_TIMEPOINTS_VEC,mean(ft_2),std(ft_2));
%         surf(X,Y,ft_2')
        yticks(0:.1:1)
        yticklabels(0:.1:1)
        xticks(NUM_TIMEPOINTS_VEC)
        xlabel('Number of Timpoints')
        ylabel('Accuracy')
        title([tvec{tf} ', Means over CV Folds'])
%         xlabel('Cross Val Folds')
    else

        plot(mse_tot(1,:))
        hold on
        plot(mse_tot(2,:))
        axis([1 FOLDS 0 1])
        xlabel('Cross Val Folds')
        ylabel('Accuracy')
    end
%     title(tvec{tf})
    %     axis([-inf inf -inf inf])
    legend('Foot L', 'Foot R')
%     xticklabels(1:FOLDS)
end
ct = {'byStride','continuous'}; md = {'LDA','CNN','LSTM'}; 
legend('Foot L', 'Foot R')
linkaxes([ax(1),ax(2)],'xy')
axis([NUM_TIMEPOINTS_VEC(1)-1 NUM_TIMEPOINTS_VEC(end)+1 0 1])
pic_title = [date '_' ct{CONTINUOUS+1} '_' md{find([LDA CNN LSTM])}];
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
