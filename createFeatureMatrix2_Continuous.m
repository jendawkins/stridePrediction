function [feature_matrix] = createFeatureMatrix2_Continuous(strideList, strideListOther, prediction_signals, two_feet, WINDOW_SIZE)


nOutputSignals = numel(prediction_signals)*(two_feet+1);

lenStrideList = numel(strideList.(prediction_signals{1}));
means = zeros(lenStrideList- WINDOW_SIZE, numel(prediction_signals)*(two_feet+1));
maxs = means;
mins = means;
ranges = means;

jrange = 1:(two_feet+1):nOutputSignals;
start = 1;

% if lenStrideList == WINDOW_SIZE
%     lenStrideList = lenStrideList+1;
% end

for i=1:lenStrideList-WINDOW_SIZE
    window_start = i; window_end = i + WINDOW_SIZE;

    iter = 1;
%     lsend = start + length(strideList(i).(prediction_signals{1}));

%     start = lsend(end);
    
    lone_iter = 1;
    delj = [];
    for jj=1:length(jrange)
        j = jrange(jj);

        means(i,j) = mean(strideList.(prediction_signals{iter})(window_start:window_end));
        maxs(i,j) = max(strideList.(prediction_signals{iter})(window_start:window_end));
        mins(i,j) = min(strideList.(prediction_signals{iter})(window_start:window_end));
        ranges(i,j) = range(strideList.(prediction_signals{iter})(window_start:window_end));

        if two_feet

            means(i,j+1) = mean(strideListOther.(prediction_signals{iter})(window_start:window_end));
            maxs(i,j+1) = max(strideListOther.(prediction_signals{iter})(window_start:window_end));
            mins(i,j+1) = min(strideListOther.(prediction_signals{iter})(window_start:window_end));
            ranges(i,j+1) = range(strideListOther.(prediction_signals{iter})(window_start:window_end));

        end
        iter = iter+1;
    end
end


%     maxs(:,delj)=[]; mins(:,delj) = []; ranges(:,delj) = [];
feature_matrix = [maxs, mins, ranges];
%     feature_matrix = [maxs(:,1:end-size(lone_sig,2)), mins(:,1:end-size(lone_sig,2)), ...
%         ranges(:,1:end-size(lone_sig,2)), lone_sig];
%     if size(feature_matrix,2)>30
%         x = 'debug';
%     end

%     rperm = randperm(lenStrideList);
%     randomized_feature_matrix = feature_matrix(rperm,:);
%
%     labels_in = lab_foot(rperm);

end