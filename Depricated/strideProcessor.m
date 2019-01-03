function [data_past, model, cycle_time, W] = strideProcessor(data_in, Fs, cycle_time,label, model, W)
% cycle time in seconds
% Data Past: time, x, y, z, gx, gy, gz, heel strike = 1; plantar flexion = 2, 
strike_dat = data_in(data_in(:,end)==1,:); 
strike_time = size(stride_dat,1);

stride_dat = data_in(data_in(:,end)==2,:);
if any(tsttime < .55*.5*cycle_time + .1 | tsttime >= .55*.5*cycle_time - .1) % 50 ms before plantar flexion -- Is this enough time??
    pt_in = [pt_in label 2];
    data_past = [data_past; pt_in];
    std_c = std(data_past(l(end):end,1:6));
    std_o = std(data_past(:,9:14));
    mean_c = mean(data_past(l(end):end,1:6));
    mean_o = mean(data_past(l(end):end,9:14));
    min_c = min(data_past(l(end):end,1:6));
    min_o = min(data_past(l(end):end,9:14));
    max_c = max(data_past(l(end):end,1:6));
    max_o = max(data_past(l(end):end,9:14));
    data_in = [data_past(l(end):end,1:end-2); pt_in(1:end-2)];
    if label == 0 
        % test
        [~, labels] = max(data_in*W);
        data_past(l(end):end,end-1)=repmat(labels,[length(l(data_in(l(end)+1:end,end))), 1]);
        pt_in(end-1) = labels;
    else
        % do LDA over all of data
        data_in = [data_past(:,1:end-2); pt_in(1:end-2)];
%         Priors calculated this way within function
%         n_classes = 3; N = sum(data_past(:,end)~=0);
%         priors = [sum(data_past(:,end-1)==1)/n_classes; sum(data_past(:,end-1)==2)/n_classes;...
%             sum(data_past(:,end-1)==3)/n_classes];
        Wnew = LDA(data_in, data_past(:,end-1));
        W = W - Wnew;
        % train
    end
end

% only check for new heel strike if old one is more than 400 ms ago 
if strike_time > .4/Fs
    pow = pt_in.^2;
    rA = sqrt(sum(pow(:,1:3),2).^2);
    thresh = prctile(data_in, 97);
    if rA>=thresh
        pt_in = [pt_in label 1];
        cycle_time = strike_time*.6 + cycle_time*.4;
    end
end

data_in(1,:) = 0;
data_in = [data_in; pt_in];
end