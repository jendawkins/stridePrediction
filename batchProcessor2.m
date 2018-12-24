function [Wr] = batchProcessor2(data_in, label, timein)
% cycle time in seconds
% Data Past: time, x, y, z, gx, gy, gz, heel strike = 1; plantar flexion = 2, 
stride_idxs2_st = find(data_in(:,end)==2);
stride_idxs2_ed = find(data_in(:,end)==3);

stride_idxs1_st = find(data_in(:,7)==2);
stride_idxs1_ed = find(data_in(:,7)==3);
% strike_dat = data_in(find(data_in(:,end)==1):end,:); 
% strike_time = abs(size(data_in,1)-strike_idxs(end));% time since strike

% stride_dat = data_in(find(data_in(:,end)==2):end,:);
% plantar flexion at .3*cycle time after strike
% if time since heel strike is greater than or equal to 150 ms before
% plantar flexion:
data_LDA = [];
lab_fin = [];
stride_idxs1_ed = [1; stride_idxs1_ed];
lab_fin = [lab_fin; label(stride_idxs1_st)];
t1a = {'Accel X','Accel Y','Accel Z','Gyro X','Gyro Y','Gyro Z'};
t1b = {'Upstairs', 'Downstairs', 'Regular'};
% plt_mat = [[lab_fin(1); lab_fin] unique(stride_idxs1_st)];
% plt_mat = sortrows(plt_mat,1)

figure;
hold on
for k = 1:length(stride_idxs1_st)
    std_c = std(data_in(stride_idxs1_ed(k):stride_idxs1_st(k),1:6));
    std_o = std(data_in(stride_idxs1_ed(k):stride_idxs1_st(k),8:13));
    mean_c = mean(data_in(stride_idxs1_ed(k):stride_idxs1_st(k),1:6));
    mean_o = mean(data_in(stride_idxs1_ed(k):stride_idxs1_st(k),8:13));
    min_c = min(data_in(stride_idxs1_ed(k):stride_idxs1_st(k),1:6));
    min_o = min(data_in(stride_idxs1_ed(k):stride_idxs1_st(k),8:13));
    max_c = max(data_in(stride_idxs1_ed(k):stride_idxs1_st(k),1:6));
    max_o = max(data_in(stride_idxs1_ed(k):stride_idxs1_st(k),8:13));
    data_LDA = [data_LDA; std_c mean_c min_c max_c std_o mean_o min_o max_o]; 
    
    plot(timein(stride_idxs1_ed(k):stride_idxs1_st(k)),data_in(stride_idxs1_ed(k):stride_idxs1_st(k),3),'g')
end

stride_idxs2_ed = [1; stride_idxs2_ed];
lab_fin = [lab_fin; label(stride_idxs2_st)];
for k = 1:length(stride_idxs2_st)
    std_c = std(data_in(stride_idxs2_ed(k):stride_idxs2_st(k),1:6));
    std_o = std(data_in(stride_idxs2_ed(k):stride_idxs2_st(k),8:13));
    mean_c = mean(data_in(stride_idxs2_ed(k):stride_idxs2_st(k),1:6));
    mean_o = mean(data_in(stride_idxs2_ed(k):stride_idxs2_st(k),8:13));
    min_c = min(data_in(stride_idxs2_ed(k):stride_idxs2_st(k),1:6));
    min_o = min(data_in(stride_idxs2_ed(k):stride_idxs2_st(k),8:13));
    max_c = max(data_in(stride_idxs2_ed(k):stride_idxs2_st(k),1:6));
    max_o = max(data_in(stride_idxs2_ed(k):stride_idxs2_st(k),8:13));
    data_LDA = [data_LDA; std_c mean_c min_c max_c std_o mean_o min_o max_o]; 
    plot(timein(stride_idxs2_ed(k):stride_idxs2_st(k)),data_in(stride_idxs2_ed(k):stride_idxs2_st(k),3),'b')
end
title('LDA input data; Foot1: green; Foot 2: blue')
Wr = LDA(data_LDA, lab_fin);

end