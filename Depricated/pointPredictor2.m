function [data_past, correct, predicted, guesses] = pointPredictor2(pt_in, pt_idx, data_past, truth, Fs, cycle_time, W, predicted, tottime)
% cycle time in seconds
% Data Past: time, x, y, z, gx, gy, gz, heel strike = 1; plantar flexion = 2, 

% actualy incorporate time
% start pred at 7 or 14 == 3
% end pred at 7 or 14 == 2

    data_in = [data_past; pt_in];
    if pt_in(:,7)==2
        W_in = W(:,:,1);
%         pts3 = find(data_in(:,7)==3);
        pts3 = find(data_in(1:end-1,7)==2);
        if ~isempty(pts3)
            start_ind = knnsearch(pt_idx - pts3, cycle_time); 
            start_stride = pts3(start_ind);
            stride_dat = data_in(start_stride:end,:);
    %         plot(linspace(0,int*length(stride_dat),length(stride_dat)),stride_dat(:,1));
            %     || strike_time2 >= (.3*cycle_time - .15)
            std_c = std(stride_dat(:,1:6));
            std_o = std(stride_dat(:,7:12));
            mean_c = mean(stride_dat(:,1:6));
            mean_o = mean(stride_dat(:,7:12));
            min_c = min(stride_dat(:,1:6));
            min_o = min(stride_dat(:,7:12));
            max_c = max(stride_dat(:,1:6));
            max_o = max(stride_dat(:,7:12));
            data_LDA = [std_c mean_c min_c max_c std_o mean_o min_o max_o]';

            [~, guesses] = max(W_in(:,1:end-1)* data_LDA + W_in(:,end));
            correct = truth==guesses;

            time_stride = tottime(start_stride:pt_idx);
            plot(time_stride,stride_dat(:,1),'g');
            hold on
            scatter(time_stride(end), stride_dat(end,1),'*g');
            %     scatter(time_stride(1) + .3*cycle_time - 15, stride_dat(pt_idx,8), '*r');
            plot(time_stride,stride_dat(:,8),'b');
            %     data_past(l(end):end,end-1)=repmat(labels,[length(l(data_in(l(end)+1:end,end))), 1]);
            %     pt_in(end-1) = labels;
            %     data_past = pt_in;
            predicted(1)=true;
        end
        %
%         data_past(1:round(cycle_time),:)=[];
    elseif pt_in(:,14)==2
        W_in= W(:,:,2);
%         pts3 = find(data_in(:,14)==3);
        pts3 = find(data_in(1:end-1,14)==2);
        if ~isempty(pts3)
            %         plot(linspace(0,int*length(stride_dat),length(stride_dat)),stride_dat(:,1));
            
            start_ind = knnsearch(pt_idx - pts3, cycle_time);
            start_stride = pts3(start_ind);
            stride_dat = data_in(start_stride:end,:);
            
            std_o = std(stride_dat(:,1:6));
            std_c = std(stride_dat(:,7:12));
            mean_o = mean(stride_dat(:,1:6));
            mean_c = mean(stride_dat(:,7:12));
            min_o = min(stride_dat(:,1:6));
            min_c = min(stride_dat(:,7:12));
            max_o = max(stride_dat(:,1:6));
            max_c = max(stride_dat(:,7:12));
            data_LDA = [std_c mean_c min_c max_c std_o mean_o min_o max_o]';
            
            [~, guesses] = max(W_in(:,1:end-1)* data_LDA + W_in(:,end));
            
            correct = truth==guesses;
            %     pt_in(end-1) = labels;
            predicted(2)=true;
            time_stride = tottime(start_stride:pt_idx);
            
            plot(time_stride,stride_dat(:,1),'g');
            hold on
            scatter(time_stride(end), stride_dat(end,8),'*b');
            %     scatter(time_stride(1) + .3*cycle_time - 15, stride_dat(pt_idx,8), '*r');
            plot(time_stride,stride_dat(:,8),'b');
            %     data_past = pt_in;
            %         data_past(1:round(cycle_time),:)=[];
        end
    end

% only check for new heel strike if old one is more than 400 ms ago 

% 
% if strike_time2 >= (.3*cycle_time - .15) && strike_time2 >= (.3*cycle_time - .15)
%     data_past = pt_in;
% end
% data_past(1,:) = [];
data_past = [data_past; pt_in];
if ~exist('correct')
    correct = 99;
    guesses = 99;
end
% if size(data_past, 1)>1000
%     data_past(1,:)=[];
% end
% disp(strike_time1)
% disp(strike_time2)
% disp(predicted)
end