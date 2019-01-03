%% Notes from oct 26 meeting

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
        [pks, ~] = findpeaks(st.(fnames{i})(:,k));
        pks_hgt = prctile(abs(pks),75);
        [pks, locsp] = findpeaks(st.(fnames{i})(:,k), 'MinPeakHeight',pks_hgt);
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
        pkst.(fnames{i}).(['f' (num2str(k))]) = pks;
        locst.(fnames{i}).(['f' (num2str(k))]) = locsp;

    end
%     if i == 2
        
end

%% get calibration points (no longer using?), xcorr 1
t1 = {'Stair1','Stair2','Ground1','Ground2'}; t2 = {'Accel','Gyro'};
% if exist('ptscal.mat')~=2
load('ptscal.mat');
    for ag = 1:2
        for k=1:length(fields(st))
            fnames = fields(st);
            if mod(k,2)==1
%             ff=figure;
%             title([t1{k}, ' ', t2{ag}])
            else
            [acor, lag] = xcorr(stfilt.(fnames{k-1})(1:5000,1),stfilt.(fnames{k})(1:5000,1));
            [~,I] = max(acor);
            lagdiff = lag(I);
            timediff(k) = lagdiff/100;
            end
%             subplot(2,1,mod(k,2)+1)
%             axis([0 20000 -inf inf])
%             hold on
%             plot(timepts.(fnames{k}), st.(fnames{k})(:,1 + (3*(ag-1))))
%             plot(timepts.(fnames{k}), st.(fnames{k})(:,2+ (3*(ag-1))))
%             plot(timepts.(fnames{k}), st.(fnames{k})(:,3+ (3*(ag-1))))
%             scatter(timepts.(fnames{k})(locst.(fnames{k}).(['f' num2str(1 + (3*(ag-1)))])), pkst.(fnames{k}).(['f' num2str(1 + (3*(ag-1)))]),'*r')
            %         scatter(timepts.(fnames{k})(locst.(fnames{k}).(['f' num2str(2 + (3*(ag-1)))])), pkst.(fnames{k}).(['f' num2str(2 + (3*(ag-1)))]),'*r')
            %         scatter(timepts.(fnames{k})(locst.(fnames{k}).(['f' num2str(3 + (3*(ag-1)))])), pkst.(fnames{k}).(['f' num2str(3 + (3*(ag-1)))]),'*r')
            if exist('ptscal.mat')~=2
                [xi,~] = getpts;%     [pks,locs] = findpeaks(B(:,2));
                ptscal(k,ag,:) = xi;
            end
            
%             close all
            % Now calibrate points
        end
    end
%     save('ptscal.mat','ptscal')
% else
%     load('ptscal.mat')
% end
% close all
%% Get points of walking section (no longer using after event selection)
t1 = {'Stair1','Stair2','Ground1','Ground2'}; t2 = {'Accel','Gyro'};
if exist('walking.mat')~=2
for ag = 1:2
    for k=1:length(fields(st))
        fnames = fields(st);
        timevec = timepts.(fnames{k});
        if mod(k,2)==1
            ff=figure;
            
            %             timevec = timepts.(fnames{k});
            %         else
            %             timevec = timepts.(fnames{k-1});
        end
        subplot(2,1,mod(k,2)+1)
        hold on
        plot(timepts.(fnames{k}), st.(fnames{k})(:,1 + (3*(ag-1))))
        plot(timepts.(fnames{k}), st.(fnames{k})(:,2+ (3*(ag-1))))
        plot(timepts.(fnames{k}), st.(fnames{k})(:,3+ (3*(ag-1))))
        title([t1{k}, ' ', t2{ag}])
        legend('x','y','z')
        if exist('walking.mat')~=2
            if ag == 1 && mod(k,2)==0
                title('Select Walking Sections')
                [xi,~] = getpts;
                walking.(fnames{k}) = xi;
            end
            save('walking.mat','walking');
        else
            load('walking.mat');
        end
        
        %         scatter(timepts.(fnames{k})(locst.(fnames{k}).(['f' num2str(1 + (3*(ag-1)))])), pkst.(fnames{k}).(['f' num2str(1 + (3*(ag-1)))]),'*r')
        %         scatter(timepts.(fnames{k})(locst.(fnames{k}).(['f' num2str(2 + (3*(ag-1)))])), pkst.(fnames{k}).(['f' num2str(2 + (3*(ag-1)))]),'*r')
        %         scatter(timepts.(fnames{k})(locst.(fnames{k}).(['f' num2str(3 + (3*(ag-1)))])), pkst.(fnames{k}).(['f' num2str(3 + (3*(ag-1)))]),'*r')
        %         [xi,yi] = getpts;
        
    end
end
else
    load('walking.mat');
end

%% Get points of walking section 2
t1a = {'Accel','Accel','Accel','Gyro','Gyro','Gyro'}; t1b = {'x','y','z','x','y','z'};
for k=1:length(fields(st))
    threshold=.001;
    sig = st.(fnames{k})./max(st.(fnames{k}));
    pow = sig.^2;
    window_width = 20;
    filt_pow = movmean(pow, window_width);
    event = filt_pow > threshold;
    
    rA = sqrt(sum(pow(:,1:3),2).^2);
%     rA_body = rA; rA_body(1:1000,:)=[];
    rA_2 = rA; rA_2(1001:end,:)=[];
    [strides,locs] = findpeaks(rA, 'MinPeakHeight', prctile(rA,97),'MinPeakDistance', 100);
    [strides2,locs2] = findpeaks(rA_2, 'MinPeakHeight', prctile(rA_2,97));

%     [strides,locs] = findpeaks(rA, 'MinPeakProminence', prctile(mean(rA),95));
    strides(locs<1000)=[];
    locs(locs < 1000)=[];
    calib_pt(k) = round(mean(locs2));
    
    strides_idx.(fnames{k}) = locs;
    win_idx.(fnames{k}) = [];
    locs2 = [calib_pt(k); locs];
    for i = 2:length(locs2)
        if abs(locs2(i-1)-locs2(i))>500
            win_idx.(fnames{k}) = [win_idx.(fnames{k}) locs2(i)];
        end
    end
    figure; 
    plot(timepts.(fnames{k}),rA); 
    hold on
    scatter(timepts.(fnames{k})(locs),strides,'*r')
    scatter(timepts.(fnames{k})(win_idx.((fnames{k}))),ones(1,length(win_idx.(fnames{k}))),'*g')
    scatter(timepts.(fnames{k})(calib_pt(k)),1,'*k')
    hold off
%     if strcmp(t1a{k}, 'Accel')
%         nchange = 9;
%     else
%         nchange = 5;
%     end
%     TF = ischange(st.(fnames{k}), 'MaxNumChanges', nchange);
%     figure;
%     for i = 1:6
% %         for k = 49:5:99
%         threshold=prctile(mean(pow(:,i)), 95);
%         filt_pow = movmean(pow(:,i), window_width);
%         event(:,i) = filt_pow > threshold;
%         subplot(3,2,i)
%         plot(timepts.(fnames{k}),st.(fnames{k})(:,i))
%         hold on
%         plot(timepts.(fnames{k}),event(:,i)*5000)
%         title(strjoin([t1a(i),t1b(i)],' '))
%     end
%     
%     figure;
%     for i = 1:6
%         hold on
%         plot(timepts.(fnames{k}),event(:,i))
%         leg_vec{i} = strjoin([t1a(i) ' ' t1b(i)]);
%     end
%     legend(leg_vec)
%     scatter(timepts.(fnames{k})(locst.(fnames{k}).f5), ones(size(locst.(fnames{k}).f5)),'*r')
%     ms = movsum(event,3)==1;
%     delms = ms(i+1)==0;
end

%% Calibrate left and right foot (improve)
for k = 1:4
%     diff12 = mean(ptscal(k-1,:,:) - ptscal(k,:,:),3);
%     timepts.(fnames{k-1})= timepts.(fnames{k-1}) - mean(diff12);
% assume calibration after 2 seconds => location 200
    
%     fploc = find(locst.(fnames{k}).f1>100);
%     fstpk = locst.(fnames{k}).f1(fploc);
%     lr = mean(ptscal(k,:,:),3);
    lr = calib_pts(k);
    % 2nd is after 1st
    fstpk = knnsearch(timepts.(fnames{k}),mean(lr));
    st.(fnames{k}) = st.(fnames{k})(fstpk:end,:);
    if mod(k,2) == 0
        walking.(fnames{k}) = walking.(fnames{k})-(timepts.(fnames{k})(fstpk)+1000);
    end
    timepts.(fnames{k}) = timepts.(fnames{k})(fstpk:end)-(timepts.(fnames{k})(fstpk)+1000);
    
end
%% Section out up/down/straight with 50 ms
fin_mat = [];
for k = 1:length(fields(walking))
    fnames = fields(walking);
    w = walking.(fnames{k});
    
    ms_time = timepts.(fnames{k});
    avg_int = mean(ms_time(2:end)-ms_time(1:end-1));
    n_500 = 50;
    
    % aoi is indices
    locs = [];
    labels = [];
    iter = 1;
    for i = 1:2:length(w)
        aoi = intersect(find(timepts.(fnames{k})>(w(i)+500)),...
            find(timepts.(fnames{k})<(w(i+1)-500)));
        n_supp = n_500 - mod(length(aoi),n_500); % make sure sections are multiples of 500
        aoi = [aoi; (aoi(end)+1:aoi(end)+n_supp)'];
        if strcmp(fnames{k},'s2')
            if mod(iter,2)~=0
                labels = [labels; repmat(1, [length(aoi),1])];
            else
                labels = [labels; repmat(2, [length(aoi),1])];
            end
        else
            labels = [labels; repmat(3, [length(aoi),1])];
        end
        aoi_r = reshape(aoi, [numel(aoi)/n_500,n_500]);
        aoi_scramb = reshape(aoi_r(randperm(numel(aoi)/n_500),:),[numel(aoi),1]);
        locs = [locs; aoi];
        iter = iter+1;
    end
    ind_rmv = locs>length(ms_time);
    locs(ind_rmv)=[];
    labels(ind_rmv)=[];
%     locs_walking.(fnames{k}) = locs;
%     locs_transition(:,k) = setdiff(1:length(timepts.(fnames{k})),locs);
%     labels(:,k) = zeros(1,length(timepts.(fnames{k})));
%     labels(locs_walking,k) = 1;
    
%     labels = repmat(k,[length(locs),1]);
    avg_int = mean(ms_time(2:end)-ms_time(1:end-1));
    n_500 = round(500/avg_int);
%     fin_time = ms_time(1):avg_int:ms_time(end);
%     sig_int = [];
%     for j = 1:6
%         sig_int(:,j) = interp1(ms_time,st.(fnames{k})(:,j),fin_time');

    fin_mat = [fin_mat; [ms_time(locs) st.(fnames{k})(locs,:) labels]];

        
end
one_mat = fin_mat(fin_mat(:,end)==1,:);
two_mat = fin_mat(fin_mat(:,end)==2,:);
thre_mat = fin_mat(fin_mat(:,end)==3,:);

%% find frequency for each category
timept = linspace(0, 10, length(fin_mat));
t1a = {'Accel','Accel','Accel','Gyro','Gyro','Gyro'}; t1b = {'x','y','z','x','y','z'};
for i = 1:6
%     f1 = figure;
% %     plot(timept,fin_mat(:,i+1))
%     plot(timept(fin_mat(:,end)==1),one_mat(:,i+1))
%     hold on
%     plot(timept(fin_mat(:,end)==2),two_mat(:,i+1))
%     plot(timept(fin_mat(:,end)==3),thre_mat(:,i+1))
% %     plot(timept,fin_mat(:,end).*1000)
%     legend('Upstairs','Downstairs','Walking')
%     title([t1a(i) t1b(i)]);
%     hold off
%     f2 = figure;
    for k = 1:3
%         f2 = figure;
%         hold on
        mat = fin_mat(fin_mat(:,end)==k,:);
        freq = fft(mat(:,i+1));
        Fs = 1/(10/1000); L = length(one_mat(:,i+1));
        P2 = abs(freq/L);
        P1 = P2(1:L/2+1);
        P1(2:end-1) = 2*P1(2:end-1);
        f = Fs*(0:(L/2))/L;
%         semilogx(f,P1)
%         title([t1a(i) t1b(i) ' ' num2str(k)])
%         hold on
        % Rows are accel: xyz, Gro: xyz
        % Columns are Upstairs, Downstairs, Walking
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
close all

%% Find peaks again using minpeakdistance
for k = 1:3
    peakdist = mode(1./maxfreq(:,k));
    
    for i = 1:6
%     mat = fin_mat(fin_mat(:,end)==k,:);
    if k == 1 || k == 2
        fname = 's1';
    else
        fname = 'g1';
    end
    [pks1, ~] = findpeaks(st.(fname)(:,i));
    pks_hgt = prctile(abs(pks1),99);
%     figure;
%     [acor, lag] = autocorr(abs(st.(fname)(:,i)),'NumLags',100);
    [pks, locsp] = findpeaks(abs(st.(fname)(:,i)), 'MinPeakHeight',pks_hgt,'MinPeakDistance',(peakdist-.1)*100);
%     [pks, locsp] = findpeaks(abs(st.(fname)(:,i)), 'MinPeakProminence',10000);
%     [pks2, locsp2] = findpeaks(st.(fname)(:,i), 'Threshold',-pks_hgt,'MinPeakDistance',peakdist);
%     pks = [pks1; pks2]; locsp = [locsp1; locsp2];
    figure; hold on
    plot(timepts.(fname), st.(fname)(:,i));
    
    if max(pks1)==max(pks)
        scatter(timepts.(fname)(locsp), pks,'*r')
    else
        scatter(timepts.(fname)(locsp), -pks,'*r')
    end
    end
    %     pkst.(fnames{i}).(['f' (num2str(k))]) = pks;
    %     locst.(fnames{i}).(['f' (num2str(k))]) = locsp;
end
% final shuffle
% rowgps = size(fin_mat,1)/n_500; rgps = randperm(rowgps);
% r=repmat(randperm(size(fin_mat,1))',1,n_500)';
% r=r(:)';
% 
% title = {'Time','AccelX1','AccelY1','AccelZ1','GX1','GY1','GZ1','Label'};
% data_fin = fin_mat(r,:);
% Filter 
% for i = 1:length(fields(st))
%     fnames = fields(st);
%     for k = 1:6
%         ms_time = timepts.(fnames{i});
%         avg_int = mean(ms_time(2:end)-ms_time(1:end-1));
%         sig_int = interp1(ms_time,st.(fnames{i})(:,k),ms_time(1):avg_int:ms_time(end));
%         freq.(fnames{i})(:,k) = fft(sig_int);
%         Fs = 1/(avg_int/1000); L = length(ms_time);
%         P2 = abs(freq.(fnames{i})(:,k)/L);
%         P1 = P2(1:L/2+1);
%         P1(2:end-1) = 2*P1(2:end-1);
%         f = Fs*(0:(L/2))/L;
% %         semilogx(f,P1) 
%     end
% end

%% practice xcorr
% [acor, lag] = xcorr(stfilt.s1(1:5000,1),stfilt.s2(1:5000,1));
% [~,I] = max(acor);
% lagdiff = lag(I);
% timediff = lagdiff/100;

