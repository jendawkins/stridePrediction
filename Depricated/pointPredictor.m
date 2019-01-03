function [data_past, correct, predicted, guesses] = pointPredictor(pt_in, pt_idx, data_past, truth, Fs, cycle_time, W, predicted, tottime)
% cycle time in seconds
% Data Past: time, x, y, z, gx, gy, gz, heel strike = 1; plantar flexion = 2, 

% actualy incorporate time
int = tottime(2)-tottime(1);

data_in = [data_past; pt_in];
stride_idxs2 = max(find(data_in(:,end)==1));
stride_idxs1 = max(find(data_in(:,7)==1));
if isempty(stride_idxs2) && isempty(stride_idxs1)
    data_past = [data_past; pt_in];
    correct = 99;
    return
end
if ~isempty(stride_idxs1)
    strike_time1 = abs(size(data_in,1)-stride_idxs1(end));% time since strike
else
    strike_time1 = -inf;
end
if ~isempty(stride_idxs2)
    strike_time2 = abs(size(data_in,1)-stride_idxs2(end));% time since strike
else
    strike_time2 = -inf;
end
% strike_dat = data_in(find(data_in(:,end)==1):end,:); 
% strike_time = abs(size(data_in,1)-strike_idxs(end));% time since strike
% stride_dat = data_in(find(data_in(:,end)==2):end,:);
% plantar flexion at .3*cycle time after strike
% if time since heel strike is greater than or equal to 150 ms before
% plantar flexion:
if strike_time1>-Inf || strike_time2>-inf

%     st = min(abs([strike_time1, strike_time2]))*int;
    % look at foot with most recent heel strike
%     if strike_time1> -inf && strike_time1<strike_time2 
%         stride_dat = data_in(round(abs(length(data_in)-cycle_time)):end,:);
%     else
%         stride_dat = data_in(round(abs(length(data_in)-cycle_time)):end,:);
%     end
% strike pt + 
    stride_dat = data_in(round(abs(length(data_in)-cycle_time)):end,:);

%     plot(tottime(st+round(.3*cycle_time - 15):size(data_in,1)),stride_dat(:,8));
%     plot(tottime(st+round(.3*cycle_time - 15):size(data_in,1)),stride_dat(:,7)*10000);
%     plot(tottime(st+round(.3*cycle_time - 15):size(data_in,1)),stride_dat(:,end)*10000);
    if strike_time1 >= (.3*cycle_time - 15) && predicted(1) == false && strike_time1<size(data_in,1) && strike_time1>-inf
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
        
        [~, guesses] = max(W(:,1:end-1)* data_LDA + W(:,end));
        correct = truth(strike_time1)==guesses;
        
        time_stride = tottime(pt_idx-(cycle_time+1):pt_idx);
        plot(time_stride,stride_dat(:,1),'g');
        hold on
        scatter(time_stride(end), stride_dat(end,1),'*g');
        %     scatter(time_stride(1) + .3*cycle_time - 15, stride_dat(pt_idx,8), '*r');
        plot(time_stride,stride_dat(:,8),'b');
        %     data_past(l(end):end,end-1)=repmat(labels,[length(l(data_in(l(end)+1:end,end))), 1]);
        %     pt_in(end-1) = labels;
        %     data_past = pt_in;
        predicted(1)=true;
        %
%         data_past(1:round(cycle_time),:)=[];
    elseif strike_time2 >= (.3*cycle_time - 15) && predicted(2) == false && strike_time2<size(data_in,1) && strike_time2>-inf
%         plot(linspace(0,int*length(stride_dat),length(stride_dat)),stride_dat(:,1));
        std_o = std(stride_dat(:,1:6));
        std_c = std(stride_dat(:,7:12));
        mean_o = mean(stride_dat(:,1:6));
        mean_c = mean(stride_dat(:,7:12));
        min_o = min(stride_dat(:,1:6));
        min_c = min(stride_dat(:,7:12));
        max_o = max(stride_dat(:,1:6));
        max_c = max(stride_dat(:,7:12));
        data_LDA = [std_c mean_c min_c max_c std_o mean_o min_o max_o]';
        
        [~, guesses] = max(W(:,1:end-1)* data_LDA + W(:,end));

        correct = truth(strike_time2)==guesses;
        %     pt_in(end-1) = labels;
        predicted(2)=true;
        time_stride = tottime(pt_idx-(cycle_time+1):pt_idx);

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
if strike_time1*int > 400 || strike_time1 == -inf
    pow = (pt_in(1:3)-mean(data_in(:,1:3))).^2;
    rA = sqrt(sum(pow));
    thresh = prctile(sqrt(sum((data_in(:,1:3)-mean(data_in(:,1:3))).^2,2)), 97);
    if rA>=thresh
%         data_in(strike_time1,7) = 1;
        cycle_time = strike_time1*.6 + cycle_time*.4;
        predicted(1)=false;
    else
        cycle_time = cycle_time;
        pt_in(7)=1;
    end
end
if strike_time2*int > 400 || strike_time2 == -inf
    pow = pt_in(8:10).^2;
    rA = sqrt(sum(pow));
    thresh = prctile(sqrt(sum((data_in(:,8:10)-mean(data_in(:,8:10))).^2,2)), 97);
    if rA>=thresh
%         data_in(strike_time2,end) = 1;
        cycle_time = strike_time2*.6 + cycle_time*.4;
        predicted(2)=false;
    else
        cycle_time = cycle_time;
        pt_in(end)=1;
    end
end
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
% disp(strike_time1)
% disp(strike_time2)
% disp(predicted)
end