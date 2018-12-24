function [data_both] = process_data_sepFeet(folder)
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


for i = 1:length(myFiles)
    if length(strfind(myFiles(i).name,'proc'))>0
        continue
    end
    x = csvread(myFiles(i).name,2,1);
    disp(myFiles(i).name)
    if i ==1
        avg_int = x(end,1)/size(x,1);
        Fs = 1/(avg_int/1000);  
    end
    fin_time = avg_int:avg_int:x(end,1);
    minlength2 = min([size(x,1), length(fin_time)]);
    fin_time = fin_time(1:minlength2)';
    x = x(1:minlength2,:);
    if size(x,1)<100
        continue
    end
    
    orig_time = x(:,1);
    x = x(:,3:8);
    x_int = [];
    for k = 1:size(x,2)
        x_int(:,k) = interp1(orig_time, x(:,k), fin_time);
        % FILTER DATA HERE!!
%         if filter_sig
%             x_int(:,k) = filter(F_h, x_int(:,k));
%             x_int(:,k) = filter(F_l, x_int(:,k));
%         end
    end
    sig = sqrt(sum((x(:,1:3)/max(x(:,1:3))).^2,2));
    sig2 = sig(1:750);
    [~, locs2] = findpeaks(sig2,"MinPeakHeight",prctile(sig2,97));
    calib_pt(i) = round(mean(locs2));
%     x_int = x_int(calib_pt(i):end,:);
%     sig = sig(calib_pt(i):end,:);
    
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
            keep_inds_bool = [keep_inds_bool; zeros([round(s_to_s),1])];
            if strfind(myFiles(2).name,'S')>0 && counter > 5
                ud = ud+1;
                counter = 0;
            end
        else
            if counter > 9 & strfind(myFiles(i).name,'S')>0
                ud = ud+1;
                counter = 0;
            end
            keep_inds_bool = [keep_inds_bool; ones([round(s_to_s),1])];
%             cycle_mat = [cycle_mat; ((1:(s_to_s+1))./s_to_s)'];
            counter = counter +1;
        end
        if strfind(myFiles(i).name,'S')>0
            lab = (mod(ud,2)+1)*ones([s_to_s,1]);
        else
            lab = 3*ones([s_to_s,1]);
            
        end
        labels = [labels; lab];
    end
    keep_inds_bool = [keep_inds_bool; zeros([size(x_int,1)-length(keep_inds_bool),1])];
    labels = [labels;zeros([size(x_int,1)-length(keep_inds_bool),1])]; 
    keep_inds_bool(locs<(locs(end)-1000))=1;
    keep_inds = find(keep_inds_bool);
    final_time = linspace(start_time, length(keep_inds)*avg_int + start_time, length(keep_inds));
    label_vec2 = [labels(keep_inds(keep_inds<length(labels))); zeros([keep_inds(end)-length(labels)+1,1])];
    data_both = [data_both; final_time' x_int(keep_inds(keep_inds<size(x_int,1)),:) strides_bool(keep_inds) label_vec2];

    start_time = final_time(end) + avg_int;
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
tvec = {'AccelX','AccelY','AccelZ','GyX','GyY','GyZ'};

for j = 1:6
    figure
    plot(data_both(:,1), data_both(:,j+1),'g')
    hold on
    scatter(data_both(find(data_both(:,8)),1), ones([sum(data_both(:,8)),1]),'*r');
   
    legend('Foot','Strides')

    title(tvec(j))
end
save('f1_processed_data.csv', 'data_both');
end