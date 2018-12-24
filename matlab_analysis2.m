%% Notes from oct 26 meeting
dbstop if error
% Cross corr function (f(x), g(x)) f x g = \sum f(t) dot g(t-h)  -- window
% over three spikes

% low pass filter! 

% Approach roman's used: stride by stride; once a stride, make
% classification; times when you're switching are predictible
% paper: translational motion tracking of joints for enhanced prediction of
% walking tasks
% use IMU data to predict postition in space of knee and ankle and classify
% based on that
% multi-class LDA

% start thinking about methods

% 3 things to optimize: 
% 1. Exploring, self interest,
% 2. Interest in lab
% 3. Publication

% MAKE QUASI PAPER: 
% Ideal case: what are your figures; what are your arguments; 

% This week: make list of these are things in the past, this is what i'm
% going to try; (data from both legs or one leg??)

close all
clear

st.s1 = csvread('2018-10-16_16h23m01s_Stair1_Foot1.csv',2,1);
st.s2 = csvread('2018-10-16_16h22m59s_Stair1_Foot2.csv',2,1);
st.g1 = csvread('2018-10-16_16h32m58s_FlatGround_Foot1.csv',2,1);
st.g2 = csvread('2018-10-16_16h32m56s_FlatGround_Foot2.csv',2,1);

d = designfilt('highpassfir', 'FilterOrder', 7, 'CutoffFrequency', 0.8);
% d2 = designfilt('lowpassfir','FilterOrder',7,'CuttoffFrequency',30);
% Filter out low freq too;
%% Filter, interpolate, findpeaks-1
% from figures: 1 == left, 2 == right
data_mat_f = [];
stride_mat_f= [];
for i = 1:length(fields(st))
    fnames = fields(st);
    timepts.(fnames{i}) = st.(fnames{i})(:,1);
    st.(fnames{i}) = st.(fnames{i})(:,3:8);    
    for k = 1:6
        ms_time = timepts.(fnames{i});
        avg_int = mean(ms_time(2:end)-ms_time(1:end-1));
        Fs = 1/(avg_int/1000);
        st.(fnames{i})(:,k) = interp1(ms_time,st.(fnames{i})(:,k),ms_time(1):avg_int:ms_time(end))';
        st.(fnames{i})(:,k) = st.(fnames{i})(:,k) - mean(st.(fnames{i})(:,k));
        % filter out gravity data: https://arxiv.org/pdf/1107.4414.pdf
        stfilt.(fnames{i})(:,k) = filter(d, st.(fnames{i})(:,k));
%         stfilt.(fnames{i})(:,k) = filter(d2, stfilt.(fnames{i})(:,k));
%         [pks, ~] = findpeaks(st.(fnames{i})(:,k));
%         pks_hgt = prctile(abs(pks),75);
%         [pks, locsp] = findpeaks(st.(fnames{i})(:,k), 'MinPeakHeight',pks_hgt);
%         figure;
%         hold on
%         plot(timepts.(fnames{k}), st.(fnames{k})(:,1 + (3*(ag-1))))
%         plot(timepts.(fnames{k}), st.(fnames{k})(:,2+ (3*(ag-1))))
%         plot(timepts.(fnames{k}), st.(fnames{k})(:,3+ (3*(ag-1))))
%         pks_cal1 = find(stfilt.(fnames{i})(locs,k)>5000);
%         % assume I do the clicking heels in the first 20 seconds
%         pks_cal2 = find(timepts.(fnames{i})(locs)<20000);
%         pks_calb = intersect(pks_cal1, pks_cal2);
%         pks_cal(i,k,:) = pks(pks_calb); locs_cal(i,k,:) = locs(pks_calb);
%         pkst.(fnames{i}).(['f' (num2str(k))]) = pks;
%         locst.(fnames{i}).(['f' (num2str(k))]) = locsp;

    end
%     if i == 2
        
end
%% Get points of walking section
fin_mat = [];
t1a = {'Accel','Accel','Accel','Gyro','Gyro','Gyro'}; t1b = {'x','y','z','x','y','z'};
for k=1:length(fields(st))
    threshold=.001;
    sig = st.(fnames{k})./max(st.(fnames{k}));
    pow = sig.^2;
    window_width = 20;
%     filt_pow = movmean(pow, window_width);
%     event = filt_pow > threshold;
    
    rA = sqrt(sum(pow(:,1:3),2));
    rA_2 = rA; rA_2(750:end,:)=[];
    [strides,locs] = findpeaks(rA, 'MinPeakHeight', prctile(rA,97),'MinPeakDistance', 100);
    [strides2,locs2] = findpeaks(rA_2, 'MinPeakHeight', prctile(rA_2,97));

    strides(locs<1000)=[];
    locs(locs < 1000)=[];
    calib_pt(k) = round(mean(locs2));
    
    strides_idx.(fnames{k}) = locs;
    win_idx.(fnames{k}) = [];
    locs2 = [calib_pt(k); locs];
    locs_bw_n = locs; locs_bw = locs;
    if mod(k,2)~=0
        if exist(['w',num2str(k),'.mat'])~=2
            figure;
            plot(timepts.(fnames{k}),rA);
            hold on
            scatter(timepts.(fnames{k})(locs),strides,'*r')
            [wi,~] = getpts;
            close
            save(['w',num2str(k)], 'wi');
%             save(['w',num2str(k+1)], 'wi');
        else
            load(['w',num2str(k)])
%             load(['w',num2str(k+1)])
        end
        win_idx.(fnames{k}) = locs(knnsearch(timepts.(fnames{k})(locs),wi));
%         win_idx.(fnames{k+1}) = locs(knnsearch(timepts.(fnames{k})(locs),wi));
    else
        stride_diff = (strides_idx.(fnames{k})(1)-calib_pt(k)) - (strides_idx.(fnames{k-1})(1)-calib_pt(k-1));
        win_idx.(fnames{k}) = win_idx.(fnames{k-1}) + calib_pt(k) - stride_diff;
    end
    
    % Calibrate left and right points
    fstpk = calib_pt(k);
    % 2nd is after 1st
%     fstpk = knnsearch(timepts.(fnames{k}),mean(lr));
    st.(fnames{k}) = st.(fnames{k})(fstpk:end,:);
    rA = rA(fstpk:end,:);
%     submat = repmat([-1,1],length(win_idx.(fnames{k}))/2);
    win_idx.(fnames{k}) = win_idx.(fnames{k})-calib_pt(k);
    timepts.(fnames{k}) = timepts.(fnames{k})(fstpk:end)-(timepts.(fnames{k})(fstpk));
    locs = locs - calib_pt(k);
    locs(locs<0) = [];
    calib_pt(k) = calib_pt(k) - calib_pt(k)+1;
    strides_idx.(fnames{k}) = locs;

    % see where Az - Wy^2*r (try this later)
%     r = .25; % estimate assuming foot-knee = 30.5% of height; accel at 1/2 foot to knee
%     foot_grnd_est = st.(fnames{k})(:,3) - r*(st.(fnames{k})(:,5)).^2;
%     n98 = intersect(find(foot_grnd_est>= -20), find(foot_grnd_est<= 0));
%     figure;
%     plot(timepts.(fnames{k}), foot_grnd_est);
%     hold on
%     plot(timepts.(fnames{k}), st.(fnames{k})(:,3));
%     scatter(timepts.(fnames{k})(n98),10*ones(length(n98),1),'*r');
%     hold off
    
    % From heel strike to push off ~ 40% of gait cycle (openSim)
    % Assume push off is half of this
    w_mat = reshape(win_idx.(fnames{k}),2,length(win_idx.(fnames{k}))/2)';
    sdiff = round((strides_idx.(fnames{k})(2:end) - strides_idx.(fnames{k})(1:end-1))*0.4*0.5, -1);
    ind_diff = sdiff>prctile(sdiff, 80);
    sdiff(ind_diff)=mode(sdiff(ind_diff==0));
    mid_stride = strides_idx.(fnames{k})(1:end-1) + sdiff;
%     mid_stride(mid_stride>w_mat(1:end-1,1) & mid_stride<w_mat(1:end-1,2)) = [];
%     figure;
%     scatter(timepts.(fnames{k})(mid_stride),ones(1,length(mid_stride))); 
%     hold on
%     plot(timepts.(fnames{k}),st.(fnames{k})(:,1)); 
%     scatter(timepts.(fnames{k})(locs),strides,'*r')
%     scatter(timepts.(fnames{k})(win_idx.(fnames{k})),ones(1,length(win_idx.(fnames{k}))),'*g')
%     scatter(timepts.(fnames{k})(calib_pt(k)),1,'*k')
    
    
    figure; 
    plot(timepts.(fnames{k}),rA); 
    hold on
    scatter(timepts.(fnames{k})(locs),strides,'*r')
    scatter(timepts.(fnames{k})(win_idx.(fnames{k})),ones(1,length(win_idx.(fnames{k}))),'*g')
    scatter(timepts.(fnames{k})(calib_pt(k)),1,'*k')
%     
%     hold off

    % Section out by total section
    
    ms_time = timepts.(fnames{k});
    avg_int = mean(ms_time(2:end)-ms_time(1:end-1));
    
    % aoi is indices
    locsf = [];
    locsf_s = [];
    labels = [];
    labels_s = [];
    iter = 1;
    
    w = win_idx.(fnames{k});
    for i = 1:2:length(w)-1
        aoi = intersect(find(timepts.(fnames{k})>(timepts.(fnames{k})(w(i)))),...
            find(timepts.(fnames{k})<(timepts.(fnames{k})(w(i+1)))));

        stride_aoi = timepts.(fnames{k})(strides_idx.(fnames{k})(timepts.(fnames{k})(strides_idx.(fnames{k}))>=(timepts.(fnames{k})(w(i))) & ...
            timepts.(fnames{k})(strides_idx.(fnames{k}))<=(timepts.(fnames{k})(w(i+1)))));
        if strcmp(fnames{k},'s2') || strcmp(fnames{k},'s1')
            if mod(iter,2)~=0
                labels = [labels; repmat(1, [length(aoi),1])];
                labels_s = [labels_s; repmat(1, [length(stride_aoi),1])];
            else
                labels = [labels; repmat(2, [length(aoi),1])];
                labels_s = [labels_s; repmat(2, [length(stride_aoi),1])];
            end
        else
            labels = [labels; repmat(3, [length(aoi),1])];
            labels_s = [labels_s; repmat(3, [length(stride_aoi),1])];
        end
        locsf = [locsf; aoi];
        locsf_s = [locsf_s; stride_aoi];
        iter = iter+1;
    end
%     ind_rmv = locsf>length(ms_time);
%     locsf(ind_rmv)=[];
%     labels(ind_rmv)=[];
    avg_int = mean(ms_time(2:end)-ms_time(1:end-1));
    if mod(k,2)==0
        labels = labels + 3;
        labels_s = labels_s + 3;
    end
    fin_mat = [fin_mat; timepts.(fnames{k})(locsf) st.(fnames{k})(locsf,:) labels];  
    stride_mat_f = [stride_mat_f; locsf_s labels_s];
%     figure;
%     plot(fin_mat(:,1),fin_mat(:,2));
end
avg_int = mean(fin_mat(2:end,1)-fin_mat(1:end-1,1));
new_time = linspace(0,length(fin_mat(:,1))*avg_int/2,length(fin_mat(:,1))/2);
strides_newidx = knnsearch(fin_mat(:,1),stride_mat_f(:,1));
st_mat = zeros([2*length(new_time),1]);
st_mat(strides_newidx) = 1;

fin_mat2 = [fin_mat(:,1:end-1) st_mat fin_mat(:,end)];

foot1 = fin_mat2(fin_mat2(:,end)<=3,:); foot1(:,end)=[]; foot1(:,1)=[];
foot2 = fin_mat2(fin_mat2(:,end)>3,:); foot2(:,end) = foot2(:,end)-3; foot2(:,1)=[];
fin_mat3 = [new_time' foot1  foot2];
labels = fin_mat3(:,end); data_in = fin_mat3(:,2:end-1);

z = find(fin_mat3(:,8)==1);
y = z(2:end)-z(1:end-1);
cycle_time = mean(y(y<prctile(y,75) & y>prctile(y,25)));
[W] = batchProcessor(data_in, labels);

start_ind = round(max([find(data_in(:,7)==1,2);find(data_in(:,end)==1,2)])/2);
data_past = data_in(1:start_ind-1,:);
correct_vec=[];
predicted = [false, false];

for pt_ind = start_ind:length(data_in)
    pt_in= data_in(pt_ind,:);
    pt_in(7)=0; pt_in(end)=0;
    [data_past, correct_ans, predicted] = pointPredictor(pt_in, data_past, labels, Fs, cycle_time, W, predicted, new_time);
    correct_vec = [correct_vec, correct_ans];
end
acc = sum(correct_vec==1)/sum(correct_vec~=99);

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

