function [data_past, correct, predicted, guesses] = pointPredictor3(pt_in, pt_idx, data_past, truth, two_feet, cycle_time, W, predicted)
% cycle time in seconds
% Data Past: time, x, y, z, gx, gy, gz, heel strike = 1; plantar flexion = 2, 

% actualy incorporate time
% start pred at 7 or 14 == 3
% end pred at 7 or 14 == 2

inverse_kinematics_outputs = {'r0','r1','r2','r3','gAccY','gAccZ','gVelY','gVelZ','gPosY','gPosZ'};

prediction_signals = ['aAccY','aAccZ','aOmegaX','d1aAccY','d1aAccZ','d1aOmegaX','i1aAccY','i1aAccZ','i1aOmegaX',inverse_kinematics_outputs];


nOutputSignals = numel(prediction_signals)*(two_feet+1);

data_in = [data_past; pt_in];
if pt_in(:,7)==2
    % %         pts3 = find(data_in(:,7)==3);
    pts3 = find(data_in(1:end-1,7)==2);
    if ~isempty(pts3)
        W_in = W.Foot1;
        start_ind = knnsearch(pt_idx - pts3, cycle_time);
        stride1 = romanFunction_1stride(data_in(start_ind:end,1:7));
        stride2 = romanFunction_1stride(data_in(start_ind:end,8:14));
        iter = 1;
        jrange = 1:(two_feet+1):nOutputSignals;
        for j=jrange
            means(j) = mean(stride1.(prediction_signals{iter}));
            maxs(j) = max(stride1.(prediction_signals{iter}));
            mins(j) = min(stride1.(prediction_signals{iter}));
            ranges(j) = range(stride1.(prediction_signals{iter}));
            if two_feet
                means(j+1) = mean(stride2.(prediction_signals{iter}));
                maxs(j+1) = max(stride2.(prediction_signals{iter}));
                mins(j+1) = min(stride2.(prediction_signals{iter}));
                ranges(j+1) = range(stride2.(prediction_signals{iter}));
            end
            iter = iter+1;
        end
        data_LDA = [maxs, mins, ranges];
        
%         L = [ones(size(data_LDA,1),1) data_LDA] * W_in';
%         P = exp(L) ./ sum(exp(L),2);
%         [~, guesses] = max(P);
        guesses = predict(W_in, data_LDA);
%         [~, guesses] = max(W_in(:,1:end-1)* data_LDA' + W_in(:,end));
        correct = truth==guesses;
    end
        %
%         data_past(1:round(cycle_time),:)=[];
elseif pt_in(:,14)==2 && two_feet
    % %         pts3 = find(data_in(:,7)==3);
    pts3 = find(data_in(1:end-1,14)==2);
    if ~isempty(pts3)
        W_in = W.Foot2;
        start_ind = knnsearch(pt_idx - pts3, cycle_time);
        stride1 = romanFunction_1stride(data_in(start_ind:end,8:13));
        stride2 = romanFunction_1stride(data_in(start_ind:end,1:6));
        iter = 1;
        jrange = 1:(two_feet+1):nOutputSignals;
        for j=jrange
            means(j) = mean(stride1.(prediction_signals{iter}));
            means(j+1) = mean(stride2.(prediction_signals{iter}));
            
            maxs(j) = max(stride1.(prediction_signals{iter}));
            maxs(j+1) = max(stride2.(prediction_signals{iter}));
            
            mins(j) = min(stride1.(prediction_signals{iter}));
            mins(j+1) = min(stride2.(prediction_signals{iter}));
            
            ranges(j) = range(stride1.(prediction_signals{iter}));
            ranges(j+1) = range(stride2.(prediction_signals{iter}));
            iter = iter+1;
        end
        data_LDA = [maxs, mins, ranges];
%         L = [ones(size(data_LDA,1),1) data_LDA] * W_in';        
%         P = exp(L) ./ sum(exp(L),2);
%         [~, guesses] = max(P);
        guesses = predict(W_in, data_LDA);
        correct = truth==guesses;
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
else
    yy = 'debuc';
end
% if size(data_past, 1)>1000
%     data_past(1,:)=[];
% end
% disp(strike_time1)
% disp(strike_time2)
% disp(predicted)
end