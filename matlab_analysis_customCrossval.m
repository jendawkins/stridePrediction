%% Notes from oct 26 meeting
clear W
% CROSSVAL in MATLAB
% ANOVA in statistical significance 
% Figure of One leg vs Two Legs!!!
% one leg vs two legs, window vs stride
% try to make bar graph next week!!!

% Use least squares regression for predicting time
% inputs are continuous features, output is number = ms to stance -- linear
% regression??

% if two legs is better, how early to make prediction??

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
% IQR = prctile(y,75)-prctile(y,25);
% z = z(y<prctile(y,75)+ IQR*1.5 & y>prctile(y,25)- IQR*1.5);
% y = y(y<prctile(y,75)+ IQR*1.5 & y>prctile(y,25)- IQR*1.5);

y2 = z2(2:end)-z2(1:end-1);
% IQR2 = prctile(y2,75)-prctile(y2,25);
% z2 = z2(y2<prctile(y2,75)+ IQR2*1.5 & y2>prctile(y2,25)- IQR*1.5);
% y2 = y2(y2<prctile(y2,75)+ IQR2*1.5 & y2>prctile(y2,25)- IQR*1.5);

stPF = round(z(1:end-1) +.3.*y -15); edPF = round(z(1:end-1) + .6*y);
stPF2 = round(z2(1:end-1) +.3*y2 -15); edPF2 = round(z2(1:end-1) + .6*y2);
labels = fin_mat3(:,end);
data_in = fin_mat3(:,[2:15]);

% Calculate angels from data???

data_in(stPF,7)=2; data_in(edPF,7)=3;
data_in(stPF2,14)=2; data_in(edPF2,14)=3;

% for iters = 1:10
%     TWO_FEET = mod(
cvIndices = crossvalind('Kfold',length(stPF),FOLDS);
accvec = [];
for ft = 1:2 % foot 1 or 2
    if ft==2
        stPF = stPF2;
    end
    cvIndices = crossvalind('Kfold',length(stPF),FOLDS);
    for cv = 1:FOLDS
        sample_strides = find(any(cvIndices == setdiff(1:FOLDS,cv),2));
        test_strides = find(cvIndices == cv);
        
        % sample_strides = randsample(length(stPF),round(.8*length(stPF)));
        % test_strides = setdiff(1:length(stPF), sample_strides);
        
        % labels_tr = labels(sample_strides);
        % labels_tst = labels(test_strides);
        
        data_train=[];
        labels_tr=[];
        for i = 1:length(sample_strides)
            ii = sample_strides(i);
            %     if ii ==length(stPF)
            if i<length(stPF)
                %         data_train = [data_train; data_in(stPF(ii):end,:)];
                %     else
                data_train = [data_train; data_in(stPF(i):stPF(i+1)-1,:)];
                labels_tr = [labels_tr; labels(stPF(i):stPF(i+1)-1)];
            end
            if ii ==1
                data_train = [data_in(1:stPF(1)-1,:); data_train];
                labels_tr = [labels(1:stPF(1)-1); labels_tr];
            elseif ii==sample_strides(end)
                data_train = [data_train; data_in(stPF(end),:)];
                labels_tr = [labels_tr; labels(stPF(end):end)];
            end
        end
        data_test=[];
        labels_tst=[];
        for k = 1:length(test_strides)-1
            kk = test_strides(k);
            %     if kk == length(stPF)
            %         data_test = [data_test; data_in(stPF(kk):end,:)];
            %     else
            if k<length(stPF)
                data_test = [data_test; data_in(stPF(k):stPF(k+1)-1,:)];
                labels_tst = [labels_tst; labels(stPF(k):stPF(k+1)-1)];
            end
            if kk ==1
                data_test = [data_in(1:stPF(1)-1,:); data_test];
                labels_tst = [labels(1:stPF(1)-1); labels_tst];
            elseif kk==test_strides(end)
                data_test = [data_test; data_in(stPF(end),:)];
                labels_tst = [labels_tst; labels(stPF(end):end)];
            end
            
        end
        %     figure;
        %     subplot(2,1,1)
        %     plot(1:size(data_train,1), data_train)
        %     title("Training Data, Testing Data")
        %     subplot(2,1,2)
        %     plot(1:size(data_test,1), data_test)
        gp_mat = {'m','c','y'};
%         for k = 1:two_feet+1
        FootSt = romanFunction(data_train);
        [randomized_feature_matrix, labels_in] = createFeatureMatrix(FootSt, labels_tr, ft, prediction_signals, TWO_FEET, foot_names);

        Mdl.(['Foot' num2str(ft)]) = fitcdiscr(randomized_feature_matrix,labels_in);
        %     W(:,:,k) = LDA(randomized_feature_matrix, labels_in);
%         end

        FootSt2 = romanFunction(data_test);

        [randomized_feature_matrix_test, labels_in_test] = createFeatureMatrix(FootSt2, labels_tst, ft, prediction_signals, TWO_FEET, foot_names);

        guess_train = predict(Mdl.(['Foot' num2str(ft)]), randomized_feature_matrix);
        guesses = predict(Mdl.(['Foot' num2str(ft)]), randomized_feature_matrix_test);
        acc = sum(guesses == labels_in_test)/length(guesses);
        C = confusion.getMatrix([labels_in_test;1; 2; 3], [guesses; 3; 1; 2]);
        
%         [acc, C] = crossval_fun2(data_train,labels_tr, data_test, labels_tst, PLOT, TWO_FEET, new_time);
        disp(['Fold ' num2str(cv)])
        disp(C)
        accv(cv) = acc;
        %     accvec.(['Foot' num2str(ft)]).(['Fold' num2str(cv)]) = acc;
        
        Cstruct.(['Foot' num2str(ft)]).(['Fold' num2str(cv)]) = C;
    end
    accvec.(['Foot' num2str(ft)]) = accv;
end
figure;
tvec = {'One Foot Training','Two Feet Training'};

plot(accvec.Foot1)
hold on
plot(accvec.Foot2)
title(tvec{TWO_FEET+1})
xlabel('Cross Val Folds')
ylabel('Testing Accuracy')
legend('Foot L', 'Foot R')

