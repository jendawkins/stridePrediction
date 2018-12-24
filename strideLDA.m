function [data_past, model, cycle_time] = strideLDA(data_past, pt_in, Fs, cycle_time,label, model)
% cycle time in seconds
% Data Past: time, x, y, z, gx, gy, gz, label?, peak?, 
k = find(data_past(:,end)==1); tsttime = abs(k(end)-(data_past(end)+1));
l = find(data_past(:,end)==2);
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
    if label == 0 
        % test
    else
        % train
    end
end

if tsttime > .4/Fs
    pow = pt_in.^2;
    rA = sqrt(sum(pow(:,1:3),2).^2);
    thresh = prctile(data_past, 97);
    if rA>=thresh
        pt_in = [pt_in label 1];
        cycle_time = tsttime*.6 + cycle_time*.4;
    end
end

data_past(1,:) = 0;
data_past = [data_past; pt_in];
end