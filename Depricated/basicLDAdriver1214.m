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

select_labels = labels(stPF);
% Calculate angels from data???

data_in(stPF,7)=2; data_in(edPF,7)=3;
data_in(stPF2,14)=2; data_in(edPF2,14)=3;

FootStrAll = romanFunction(data_in);
% for iters = 1:10
%     TWO_FEET = mod(
for tf = 1:2
    TWO_FEET = tf-1;
accvec = [];
for ft = 1:2 % foot 1 or 2
    if ft==2
        stPF = stPF2;
    end
    
    stridesAll = FootStrAll.([foot_names{ft}]);
    stridesAllOther = FootStrAll.([foot_names{setdiff(1:2,ft)}]);
    
    select_labels = labels(stPF);
    select_labels = select_labels(1:end-1); %***have labels as previous startPF
    cvIndices = crossvalind('Kfold',length(stPF)-1,FOLDS, 'Min',3);
    for cv = 1:FOLDS
        sample_strides = find(any(cvIndices == setdiff(1:FOLDS,cv),2));
        test_strides = find(cvIndices == cv);
        
        tslab = select_labels(test_strides);
        trlab = select_labels(sample_strides);

        ftmat_train = createFeatureMatrix2(stridesAll(sample_strides), stridesAllOther(sample_strides), prediction_signals, TWO_FEET);
        
        rperm = randperm(size(ftmat_train,1));
        ftmat_train = ftmat_train(rperm,:);
        tr_labels = trlab(rperm);
        
        
        Mdl = fitcdiscr(ftmat_train, tr_labels);
        
        ftmat_test = createFeatureMatrix2(stridesAll(test_strides), stridesAllOther(sample_strides), prediction_signals, TWO_FEET);
        guesses = predict(Mdl, ftmat_test);
        
        acc = sum(guesses == tslab)/length(guesses);
        
        C = confusion.getMatrix([tslab;1; 2; 3], [guesses; 3; 1; 2]);
        
        disp(['Fold ' num2str(cv)])
        disp(C)
        accv(cv) = acc;
        
        Cstruct.(['Foot' num2str(ft)]).(['Fold' num2str(cv)]) = C;

        
       
    end
    accvec.(['Foot' num2str(ft)]) = accv;
end
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
end

