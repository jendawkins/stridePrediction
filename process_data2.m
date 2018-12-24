function [data_both] = process_data2(folder)
close all
disp('reading data');
myFiles = dir(fullfile(folder,'*.csv'));
Fs = 1/(10/1000);
start_time = 0;
cycle_mat = [];
data_fin = [];
strides_bool_fin = [];
labels = [];
keep_inds = [];
data_both = [];

filter_sig = true;
F_lowpass  = 8;
F_highpass = .5;

it = 0;
data_noise = [];
data_noise_mean = [];
data_sig = [];
for i = 1:length(myFiles)
    if length(strfind(myFiles(i).name,'proc'))>0 | length(strfind(myFiles(i).name,'2018'))==0
        continue
    end
    x = csvread(myFiles(i).name,2,1);

    it = it +1;
    disp(myFiles(i).name)
    
    if it ==1
        avg_int = x(end,1)/size(x,1);
        Fs = 1/(avg_int/1000);
        
%         LP_IIR = dsp.LowpassFilter('SampleRate',Fs,'FilterType','IIR',...
%             'DesignForMinimumOrder',false,'FilterOrder',7,...
%             'PassbandFrequency',F_lowpass);
%         HP_IIR = dsp.HighpassFilter('SampleRate',Fs,'FilterType','IIR',...
%             'DesignForMinimumOrder',false,'FilterOrder',7,...
%             'PassbandFrequency',F_highpass);
%        dl = fdesign.lowpass('Fp,Fst,Ap,Ast',8,10,0.5,40,Fs);
%        F_l = design(dl,'equiripple');
%        dh = fdesign.highpass('Fp,Fst,Ap,Ast',.5,.3,0.5,40,Fs);
%        F_h = design(dh,'equiripple');       
    end
    fin_time = avg_int:avg_int:x(end,1);
    minlength2 = min([size(x,1), length(fin_time)]);
    fin_time = fin_time(1:minlength2)';
    x = x(1:minlength2,:);
    
    orig_time = x(:,1);
    x = x(:,3:8);
    x_int = [];
    ps = [];
    for k = 1:size(x,2)
        x_int(:,k) = interp1(orig_time, x(:,k), fin_time);
%         x_i = x_int(2:end-1,k);
%         try
%             x_filt(:,k) = highpass(x_i, F_lowpass, Fs, 'ImpulseResponse', 'iir');
%         catch
%             xx = 1;
%         end
%             % FILTER DATA HERE!!
% %         if filter_sig
% %             x_int(:,k) = filter(F_h, x_int(:,k));
% %             x_int(:,k) = filter(F_l, x_int(:,k));
% %         end
    end
    sig = sqrt(sum((x(:,1:3)/max(x(:,1:3))).^2,2));
    sig2 = sig(1:750);
    [~, locs2] = findpeaks(sig2,"MinPeakHeight",prctile(sig2,97));
    calib_pt(i) = round(mean(locs2));
    x_int = x_int(calib_pt(i):end,:);
    sig = sig(calib_pt(i):end,:);
    
    if strfind(myFiles(i).name,'S')>0
        pk_perc = 50;
        pk_dist = 130;
    else
        pk_perc = 90; 
        pk_dist = 90;
    end
    [strides, locs] = findpeaks(sig,"MinPeakHeight",prctile(sig,pk_perc),"MinPeakDistance",pk_dist);
    
    strides = strides(locs>1000);
    locs = locs(locs>1000);
    
    strides_bool = zeros([length(fin_time),1]);
    strides_bool(locs) = 1;
    
    keep_inds_old = keep_inds;
    keep_inds_bool = zeros([locs(1),1]);
    ud = 1;
    labels_old = labels;
    labels = [];
    tots = locs(2:end)-locs(1:end-1);
    labels = zeros([locs(1),1]);
    counter = 0;
    for l =1:length(locs(2:end))
%         if l > 29
%             xy = 1;
%         end
        s_to_s = tots(l);

        IQR = prctile(tots,75)-prctile(tots,25);
        if s_to_s > prctile(tots,75)+ IQR*1.5
            keep_inds_bool = [keep_inds_bool; zeros([round(s_to_s)+1,1])];
            if strfind(myFiles(2).name,'S')>0 && counter > 5
                ud = ud+1;
                counter = 0;
            end
        else
            if counter > 9 & strfind(myFiles(i).name,'S')>0
                ud = ud+1;
                counter = 0;
            end
            keep_inds_bool = [keep_inds_bool; ones([round(s_to_s)+1,1])];
%             cycle_mat = [cycle_mat; ((1:(s_to_s+1))./s_to_s)'];
            counter = counter +1;
        end
%         if strfind(myFiles(i).name,'S')>0
%             lab = (mod(ud,2)+1)*ones([s_to_s,1]);
%         else
%             lab = 3*ones([s_to_s,1]);
%             
%         end
%         labels = [labels; lab];
    end
    keep_inds_bool = [keep_inds_bool; zeros([size(x_int,1)-length(keep_inds_bool),1])];
%     labels = [labels;zeros([size(x_int,1)-length(keep_inds_bool),1])]; 
    keep_inds_bool(locs<(locs(end)-1000))=1;
    keep_inds = find(keep_inds_bool);
    
    
    
    if mod(it,2)==0
%         kp_tot = unique([keep_inds; keep_inds_old]);
        kp_tot = intersect(keep_inds, keep_inds_old);
        final_time = linspace(start_time, length(kp_tot)*avg_int + start_time, length(kp_tot));
        
%         st_label1 = find(strides_struct.(['st' num2str(i-1)]));
%         label_vec1 = [labels_old(kp_tot(kp_tot<length(labels_old))); zeros([kp_tot(end)-length(labels_old)+1,1])];
%         label_vec2 = [labels(kp_tot(kp_tot<length(labels))); zeros([kp_tot(end)-length(labels)+1,1])];
%         label_vec1(st_label1(1:end-1)) = labels_old;
        
%         st_label2 = find(strides_bool);
%         label_vec2 = zeros([length(kp_tot),1]);
%         label_vec2(st_label2(1:end-1)) = labels;
        % final data: time; foot 1 (6); strides1; labels1; foot2 (6);
        % strides2; labels2
        d1 = data_struct.(['st' num2str(i-1)])(kp_tot,:);
        d2 = x_int(kp_tot,:);
        
%         sig_k = sqrt(sum((d1(:,1:3)./max(d1(:,1:3))).^2,2));
        sig_k = sqrt(sum(d1(:,1:3).^2,2));
        [~, locs] = findpeaks(sig_k,"MinPeakHeight",prctile(sig_k,pk_perc),"MinPeakDistance",pk_dist);

        strides_bool1 = zeros([length(final_time),1]);
        strides_bool1(locs) = 1;
        sig1 = sig_k;
%         figure;
%         plot(sig_k);
%         hold on
%         t = 1:length(sig_k);
%         scatter(t(locs), sig_k(locs));
        
        sig_k = sqrt(sum(d2(:,1:3).^2,2));
        [~, locs] = findpeaks(sig_k,"MinPeakHeight",prctile(sig_k,pk_perc),"MinPeakDistance",pk_dist);
        sig2 = sig_k;
        
%         figure;
%         plot(sig_k);
%         hold on
%         t = 1:length(sig_k);
%         scatter(t(locs), sig_k(locs));
        
%         pause(2)
%         close all
        strides_bool2 = zeros([length(final_time),1]);
        strides_bool2(locs) = 1;
    
        data_sig = [data_sig; sig1 sig2];
        d1 = (d1 - mean(d1))./std(d1);
        d2 = (d2 - mean(d2))./std(d2);
        data_both = [data_both; final_time' d1 strides_bool1 d2...
            strides_bool2];
        start_time = final_time(end) + avg_int;
        
%         data_noise = [data_noise; x_filt];
%         data_noise_mean = [data_noise_mean; movmean(x_filt, 10)];
    end

    keep_inds = find(keep_inds);
    
%     x_int = [fin_time x_int];
%     data = x_int(keep_inds,:);
    data_struct.(['st' num2str(i)]) = x_int;
%     data_fin = [data_fin; data];
    strides_struct.(['st' num2str(i)]) = strides_bool;
%     strides_bool = strides_bool(keep_inds);
%     strides_bool_fin = [strides_bool_fin; strides_bool];
    
%     data_fin(:,1) = linspace(start_time,size(data_fin,1)*avg_int + start_time, size(data_fin,1));
end
data_both([2:7,9:14]) = (data_both([2:7,9:14])- mean(data_both([2:7,9:14])))./max(data_both([2:7,9:14]));
tvec = ['AccelX','AccelY','AccelZ','GyX','GyY','GyZ'];
% figure;
% plot(data_noise_mean(:,1))
if exist('wrong2.mat')~=2
figure
dat_plot = data_both(:,[2:4, 9:11]);
r1 = sqrt(dat_plot(:,1).^2 + dat_plot(:,2).^2  + dat_plot(:,3).^2);
r2 = sqrt(dat_plot(:,4).^2 + dat_plot(:,5).^2  + dat_plot(:,6).^2);
t = 1:length(r1);
plot(t, data_both(:,2),'g')
hold on
plot(t, data_both(:,9),'b')
% hold on
% t = 1:length(data_sig(:,1));
scatter(t(data_both(:,8)==1),zeros([sum(data_both(:,8)==1),1]),'*r')
scatter(t(data_both(:,15)==1),zeros([sum(data_both(:,15)==1),1]),'*y')

title('Pick Upstairs')
[pind1u,xs,ys] = selectdata('selectionmode','lasso');

title('Pick Downstairs')
[pind1d,xs,ys] = selectdata('selectionmode','lasso');

title('Pick Flat')
[pind1f,xs,ys] = selectdata('selectionmode','lasso');

stbool1 = find(data_both(:,8)==1);
labs1_bool = zeros([size(data_both,1),1]);
labs1_bool(stbool1(pind1u{2})) = ones(size(pind1u{2}));
labs1_bool(stbool1(pind1d{2})) = 2*ones(size(pind1d{2}));
labs1_bool(stbool1(pind1f{2})) = 3*ones(size(pind1f{2}));

stbool2 = find(data_both(:,end)==1);
labs2_bool = zeros([size(data_both,1),1]);
labs2_bool(stbool2(pind1u{1})) = ones(size(pind1u{1}));
labs2_bool(stbool2(pind1d{1})) = 2*ones(size(pind1d{1}));
labs2_bool(stbool2(pind1f{1})) = 3*ones(size(pind1f{1}));
% 
% 
% figure;
% plot(t, data_both(:,9),'g')
% hold on
% % t = 1:length(data_sig(:,2));
% scatter(t(data_both(:,15)==1),zeros([sum(data_both(:,15)==1),1]),'*r')
% 
% title('Pick Upstairs')
% [pind2u,xs,ys] = selectdata('selectionmode','lasso','Action','delete');
% 
% title('Pick Downstairs')
% [pind2d,xs,ys] = selectdata('selectionmode','lasso','Action','delete');
% 
% title('Pick Flat')
% [pind2f,xs,ys] = selectdata('selectionmode','lasso','Action','delete');
% 
% labs2_bool = zeros([size(data_both,1),1]);
% labs2_bool(pind2u{1}) = ones(size(pind2u{1}));
% labs2_bool(pind2d{1}) = 2*ones(size(pind2d{1}));
% labs2_bool(pind2f{1}) = 3*ones(size(pind2f{1}));

data_both = [data_both(:,1:8) labs1_bool data_both(:,9:end) labs2_bool];
% data_both = [data_both(:,1:8) labs1_bool data_both(:,11:17) labs2_bool];

for j = 1:6
    figure
    plot(data_both(:,1), data_both(:,j+1),'g')
    hold on
    scatter(data_both(find(data_both(:,8)),1), data_both(find(data_both(:,8)),9),'*r')

    plot(data_both(:,1), data_both(:,j+9),'b')
    scatter(data_both(find(data_both(:,16)),1), data_both(find(data_both(:,16)),17),'*k')
    legend('Foot 1','Strides/Labels 1','Foot 2', 'Strides/Labels 2')

    title(tvec(j))
    
    saveas(gcf, ['Figures/processed_data/' date '_' tvec(j)]);
end

if exist([date '-processed_data.csv'])~=2
    csvwrite([date '-processed_data.csv'], data_both)
end
end