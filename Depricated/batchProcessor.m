function [Wr] = batchProcessor(data_in, label)
% cycle time in seconds
% Data Past: time, x, y, z, gx, gy, gz, heel strike = 1; plantar flexion = 2, 
stride_idxs2 = find(data_in(:,end)==1);
stride_idxs1 = find(data_in(:,7)==1);
% strike_dat = data_in(find(data_in(:,end)==1):end,:); 
% strike_time = abs(size(data_in,1)-strike_idxs(end));% time since strike

% stride_dat = data_in(find(data_in(:,end)==2):end,:);
% plantar flexion at .3*cycle time after strike
% if time since heel strike is greater than or equal to 150 ms before
% plantar flexion:
data_LDA = [];
lab_fin = [];
stride_idxs1 = [1; stride_idxs1];
lab_fin = [lab_fin; label(stride_idxs1(2:end))];
t1a = {'Accel X','Accel Y','Accel Z','Gyro X','Gyro Y','Gyro Z'};
t1b = {'Upstairs', 'Downstairs', 'Regular'};
plt_mat = [[lab_fin(1); lab_fin] unique(stride_idxs1)];
plt_mat = sortrows(plt_mat,1);
% for j = 1:6
%     iter = 0;
%     for k = 2:length(stride_idxs1)
%         if plt_mat(k,1) ~= plt_mat(k-1,1) || iter==0
%             figure;
%             hold on;
%             axis([0,1500,-inf, inf])
%             title([t1b(plt_mat(k,1)) ', ' t1a(j)])
%             xlabel('Time, ms')
%         end
%         iter = iter+1;
%         hold on
%         plot(1:10:(length(plt_mat(k-1,2):plt_mat(k,2))*10), data_in(plt_mat(k-1,2):plt_mat(k,2),j))
%     end
% end
for k = 2:length(stride_idxs1)
    std_c = std(data_in(stride_idxs1(k-1):stride_idxs1(k),1:6));
    std_o = std(data_in(stride_idxs1(k-1):stride_idxs1(k),8:13));
    mean_c = mean(data_in(stride_idxs1(k-1):stride_idxs1(k),1:6));
    mean_o = mean(data_in(stride_idxs1(k-1):stride_idxs1(k),8:13));
    min_c = min(data_in(stride_idxs1(k-1):stride_idxs1(k),1:6));
    min_o = min(data_in(stride_idxs1(k-1):stride_idxs1(k),8:13));
    max_c = max(data_in(stride_idxs1(k-1):stride_idxs1(k),1:6));
    max_o = max(data_in(stride_idxs1(k-1):stride_idxs1(k),8:13));
    data_LDA = [data_LDA; std_c mean_c min_c max_c std_o mean_o min_o max_o]; 
end

stride_idxs2 = [1; stride_idxs2];
lab_fin = [lab_fin; label(stride_idxs2(2:end))];
for k = 2:length(stride_idxs2)
    std_c = std(data_in(stride_idxs2(k-1):stride_idxs2(k),1:6));
    std_o = std(data_in(stride_idxs2(k-1):stride_idxs2(k),8:13));
    mean_c = mean(data_in(stride_idxs2(k-1):stride_idxs2(k),1:6));
    mean_o = mean(data_in(stride_idxs2(k-1):stride_idxs2(k),8:13));
    min_c = min(data_in(stride_idxs2(k-1):stride_idxs2(k),1:6));
    min_o = min(data_in(stride_idxs2(k-1):stride_idxs2(k),8:13));
    max_c = max(data_in(stride_idxs2(k-1):stride_idxs2(k),1:6));
    max_o = max(data_in(stride_idxs2(k-1):stride_idxs2(k),8:13));
    data_LDA = [data_LDA; std_c mean_c min_c max_c std_o mean_o min_o max_o]; 
end
Wr = LDA(data_LDA, lab_fin);
    
% % if strike_time >= (.3*cycle_time - .15) 
% %     data_in(end,end) = 2;
% %     
% %     std_c = std(stride_dat(:,1:6));
% %     std_o = std(stride_dat(:,7:12));
% %     mean_c = mean(stride_dat(:,1:6));
% %     mean_o = mean(stride_dat(:,7:12));
% %     min_c = min(stride_dat(:,1:6));
% %     min_o = min(stride_dat(:,7:12));
% %     max_c = max(stride_dat(:,1:6));
% %     max_o = max(stride_dat(:,7:12));
% %     data_LDA = [std_c mean_c min_c max_c std_o mean_o min_o max_o]; 
%     if all(label == 0) % test
%         [~, labels] = max(data_in*W);
% %         data_past(l(end):end,end-1)=repmat(labels,[length(l(data_in(l(end)+1:end,end))), 1]);
% %         pt_in(end-1) = labels;
%     else
%         % do LDA over all of data
% %         Priors calculated this way within function
% %         n_classes = 3; N = sum(data_past(:,end)~=0);
% %         priors = [sum(data_past(:,end-1)==1)/n_classes; sum(data_past(:,end-1)==2)/n_classes;...
% %             sum(data_past(:,end-1)==3)/n_classes];
%         Wr = LDA(data_LDA, label);
%         % train
%     end

% only check for new heel strike if old one is more than 400 ms ago 
% if strike_time > .4/Fs
%     pow = pt_in.^2;
%     rA = sqrt(sum(pow(:,1:3),2).^2);
%     thresh = prctile(data_in, 97);
%     if rA>=thresh
%         data_in(end,end) = 1;
%         cycle_time = strike_time*.6 + cycle_time*.4;
%     end
% end
% 
% data_in(1,:) = 0;
% data_in = [data_in; pt_in];
end