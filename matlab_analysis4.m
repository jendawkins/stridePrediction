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
two_feet = 0;

dbstop if error
close all
if exist('data_fin')==0
    [data_fin] = process_data(pwd);
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
labels = fin_mat3(:,17);
data_in = fin_mat3(:,[2:8,10:16]);
data_in(stPF,7)=2; data_in(edPF,7)=3;
data_in(stPF2,14)=2; data_in(edPF2,14)=3;

sample_strides = randsample(length(stPF),round(.8*length(stPF)));
test_strides = setdiff(1:length(stPF), sample_strides);

labels_tr = labels(sample_strides);
labels_tst = labels(test_strides);
data_train=[];
for i = 1:length(sample_strides)
    ii = sample_strides(i);
    %     if ii ==length(stPF)
    if ii<length(stPF)
        %         data_train = [data_train; data_in(stPF(ii):end,:)];
        %     else
        data_train = [data_train; data_in(stPF(ii):stPF(ii+1)-1,:)];
    end
    if ii ==1
        data_train = [data_in(1:stPF(1)-1,:); data_train];
    elseif ii==length(stPF)
        data_train = [data_train; data_in(stPF(end),:)];
    end
end
data_test=[];
for k = 1:length(test_strides)-1
    kk = sample_strides(k);
%     if kk == length(stPF)
%         data_test = [data_test; data_in(stPF(kk):end,:)];
%     else
    if kk<length(stPF)
        data_test = [data_test; data_in(stPF(kk):stPF(kk+1)-1,:)];
    end
    if kk ==1
        data_test = [data_in(1:stPF(1)-1,:); data_test];
    elseif kk==length(stPF)
        data_test = [data_test; data_in(stPF(end),:)];
    end
        
end

cycle_time = mean(y(y<prctile(y,75) & y>prctile(y,25)));
[FootSt] = romanFunction(data_train);
inverse_kinematics_outputs = {'r0','r1','r2','r3','gAccY','gAccZ','gVelY','gVelZ','gPosY','gPosZ'};

prediction_signals = ['aAccY','aAccZ','aOmegaX','d1aAccY','d1aAccZ','d1aOmegaX','i1aAccY','i1aAccZ','i1aOmegaX',inverse_kinematics_outputs];
foot_names= {'F1','F2'};
feature_matrix =[];
lab_both = [];

for k = 1:two_feet+1
    strideList = FootSt.(foot_names{k});
    n2= setdiff(foot_names,foot_names{k});
    strideListOther = FootSt.(n2{:});
    lenStrideList = length(strideList);
    
    lab_foot = labels([strideList.globalInitStanceSample]);
    disp('Extracting predictor features...');
    nOutputSignals = numel(prediction_signals)*(two_feet+1);
    
    means = zeros(lenStrideList, numel(prediction_signals)*(two_feet+1));
    maxs = means;
    mins = means;
    ranges = means;
    
    jrange = 1:(two_feet+1):nOutputSignals;
    for i=1:lenStrideList
%         window_start = strideList(i).globalFootStaticSample - strideList(i).globalInitStanceSample + 1;
%         window_end = strideList(i).globalInitSwingSample - strideList(i).globalInitStanceSample + POST_SWING_CUTOFF_SAMPLES;
        iter = 1;
        for jj=1:length(jrange)
            j = jrange(jj);
            means(i,j) = mean(strideList(i).(prediction_signals{iter}));
            maxs(i,j) = max(strideList(i).(prediction_signals{iter}));
            mins(i,j) = min(strideList(i).(prediction_signals{iter}));            
            ranges(i,j) = range(strideList(i).(prediction_signals{iter}));
            
            if two_feet
                means(i,j+1) = mean(strideListOther(i).(prediction_signals{iter}));
                maxs(i,j+1) = max(strideListOther(i).(prediction_signals{iter}));
                mins(i,j+1) = min(strideListOther(i).(prediction_signals{iter}));
                ranges(i,j+1) = range(strideListOther(i).(prediction_signals{iter}));
            end
            iter = iter+1;
        end
    end
    feature_matrix = [maxs, mins, ranges];
    
    rperm = randperm(lenStrideList);
    randomized_feature_matrix = feature_matrix(rperm,:);

    labels_in = lab_foot(rperm);
    
    W(:,:,k) = LDA(randomized_feature_matrix, labels_in);
end
% nOutputFeatures = nOutputSignals * 3;
% [W] = batchProcessor3(data_in, labels, new_time);

% start_ind = round(max([find(data_in(:,7)==1,1);find(data_in(:,end)==1,1)]))+1;
% data_past = data_in(1:start_ind-1,:);
correct_vec=[];
predicted = [false, false];
guess_vec = [];
Fs = 1/(mean(data_fin(2:end,1)-data_fin(1:end-1,1))/1000);
figure; hold on; plot(new_time, data_fin(:,2)); plot(new_time, data_fin(:,17)*1000)

% figure;

data_past = [];
for pt_ind = 1:length(data_test)
    pt_in= data_test(pt_ind,:);
%     pt_in(7)=0; pt_in(end)=0;
    [data_past, correct_ans, predicted, guesses] = pointPredictor3(pt_in, pt_ind, data_past, labels_tst(pt_ind), two_feet, cycle_time, W, predicted, new_time);

    hold on
    correct_vec = [correct_vec, correct_ans];
    guess_vec = [guess_vec, guesses];
end
acc = sum(correct_vec==1)/sum(correct_vec~=99);
C = confusion.getMatrix([labels_tst(guess_vec~=99);1; 2; 3], [guess_vec(guess_vec~=99)'; 3; 1; 2]);

csvwrite('processed_data.csv',[data_fin(:,1) data_in labels]);
figure;
plot(new_time,fin_mat(:,2));
hold on
plot(new_time, 1000.*fin_mat(:,end))
scatter(new_time(strides_newidx), stride_mat_f(:,2))

one_mat = fin_mat(fin_mat(:,end)==1,:);
two_mat = fin_mat(fin_mat(:,end)==2,:);
thre_mat = fin_mat(fin_mat(:,end)==3,:);

%% find frequency for each category
timept = linspace(0, 10, length(fin_mat));
t1a = {'Accel','Accel','Accel','Gyro','Gyro','Gyro'}; t1b = {'x','y','z','x','y','z'};
tic
for i = 1:6
    f1 = figure;
    plot(timept,fin_mat(:,i+1))
%     plot(timept(fin_mat(:,end)==1),one_mat(:,i+1))
    hold on
%     plot(timept(fin_mat(:,end)==2),two_mat(:,i+1))
%     plot(timept(fin_mat(:,end)==3),thre_mat(:,i+1))
    plot(timept,fin_mat(:,end).*1000)
%     legend('Upstairs','Downstairs','Walking')
    title([t1a(i) t1b(i)]);
    xlabel('Time, ms')
%     hold off
%     f2 = figure;
    for k = 1:3
%         f2 = figure;
%         hold on
        mat = fin_mat(fin_mat(:,end)==k,:);
%         tic
        freq = fft(mat(:,i+1));
%         toc
        Fs = 1/(10/1000); L = length(one_mat(:,i+1));
        P2 = abs(freq/L);
        P1 = P2(1:L/2+1);
        P1(2:end-1) = 2*P1(2:end-1);
        f = Fs*(0:(L/2))/L;
        semilogx(f,P1)
        title([t1a(i) t1b(i) ' ' num2str(k)])
        hold on
        Rows are accel: xyz, Gro: xyz
        Columns are Upstairs, Downstairs, Walking
        P1(1) = 0;
        [~,mfloc] = max(P1);
        maxfreq(i,k) = f(mfloc);
        
    end
    
%     legend('Upstairs','Downstairs','Regular')
    path = pwd;
%     title([t1a(i) t1b(i)])
%     hold off
%     saveas(f1, [pwd '/Figures/' strjoin([t1a(i),t1b(i)],'_') '.png'])
%     saveas(f1, [pwd '/Figures/Color_' strjoin([t1a(i),t1b(i)],'_')])
%     saveas(f2, [pwd '/Figures/Freq_' strjoin([t1a(i),t1b(i)],'_') '.png'])
%     saveas(f2, [pwd '/Figures/Freq_' strjoin([t1a(i),t1b(i)],'_')])
end
toc
close all

