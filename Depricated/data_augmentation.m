%% Notes from oct 26 meeting

% flops: layers * timesteps * 8 * 2 * hiddenSize * minibatch * (hiddenSize + 1)
% layers = 1 * NUM_TIMEPOINTS * 8 * 2 * 
clear all

% Define parameters
POST_SWING_CUTOFF_TIME_S=0.2;
SAMPLE_RATE_HZ = 100;
POST_SWING_CUTOFF_SAMPLES = int32(POST_SWING_CUTOFF_TIME_S * SAMPLE_RATE_HZ);

FOLDS = 5;
PLOT = false;
% USE_PREV_TIME = 1;
PREDICTION_WINDOW_START = .6;
PREDICTION_WINDOW_END = 1;
NUM_TIMEPOINTS_VEC = [5 9 13]; % set to 1 to not use averaging; make vector to check multiple timepoints
CNN = 0;
LDA = 0;
LSTM=1;
plot_pts = 0;
if length(NUM_TIMEPOINTS_VEC) > 1
    CONTINUOUS = 1;
else
    CONTINUOUS = 0;
end
WINDOW_START = .6;
WINDOW_END = 1;

raw_sensor_outputs = {'a1Raw','a2Raw','a3Raw','g1Raw','g2Raw','g3Raw'};
inverse_kinematics_outputs = {'r0','r1','r2','r3','gAccY','gAccZ','gVelY','gVelZ','gPosY','gPosZ'};
derivative_outputs = {'d1aAccX','d1aAccY','d1aAccZ','d1aOmegaX','d1aOmegaY','d1aOmegaZ'};
integral_outputs = {'i1aAccX','i1aAccY','i1aAccZ','i1aOmegaX','i1aOmegaY','i1aOmegaZ'};
prediction_signals = {'aAccY','aAccZ','aOmegaX','d1aAccY','d1aAccZ','d1aOmegaX','i1aAccY','i1aAccZ','i1aOmegaX'};
if CONTINUOUS
    prediction_signals = setdiff(prediction_signals, 'HSmag');
end

foot_names= {'F1','F2'};

prediction_signals = raw_sensor_outputs;
dbstop if error
close all
dt = '22-Dec-2018';
dbstop if error
close all
% load data
if exist([dt '-processed_data 3.csv'])==0
    [data_fin] = process_data2([pwd '/Raw Data']);
else
    load([dt '-processed_data 3.csv'])
    data_fin = X22_Dec_2018_processed_data_3;
end
new_time = data_fin(:,1);
fin_mat3 = data_fin;
z = find(fin_mat3(:,8)==1);
z2 = find(fin_mat3(:,end-1)==1);

% fix discrepancy between feet

y = z(2:end)-z(1:end-1);
IQR = prctile(y,75)-prctile(y,25);

y2 = z2(2:end)-z2(1:end-1);

labels1o = fin_mat3(:,9);
labels1 = labels1o(z);
labels2o = fin_mat3(:,end);
labels2 = labels2o(z2);

delstr1 = find(y<prctile(y,25)-IQR*1.5);
delstr2 = find(y2<prctile(y2,25)-IQR*1.5);

z2(delstr2)=[];
z(delstr1)=[];
y(delstr1) = [];
y2(delstr2)=[];
labels1(delstr1) = [];
labels2(delstr2) = [];

zest2 = round(z(1:end-1) + .6*y);
zest1 = round(z2(1:end-1) + .6*y2);

z = [z(1); zest2(1); z(2:end)];
labels1 = [labels1(1); labels2o(zest2(1)); labels1(2:end)];
for j = 2:(length(z)-1)
    if z2(j)>z(j) && z2(j+1)<z(j+1) % gap in green
        %         continue
        %     else
        labest = mean([labels2o(z2(j)) labels2o(z2(j+1))]);
        if isinteger(labest)==0
            labest = 0;
        end
        zest = round(z2(j) + (z2(j+1)-z(j))*.6);
        z = [z(1:j); zest; z(j+1:end)];
        labels1 = [labels1(1:j); labest; labels1(j+1:end)];
        %         z2 = [z2(1:j); zest2(j); z2(j+1:end)];
    end
    % gap in red
    if z(j)>z2(j-1) && z(j+1)<z2(j)
        %         continue
        %     else
        labest = mean([labels1o(z(j)) labels1o(z(j+1))]);
        if isinteger(labest)==0
            labest = 0;
        end
        zest = round(z(j) + (z(j+1)-z(j))*.6);
        z2 = [z2(1:j-1); zest; z2(j:end)];
        labels2 = [labels2(1:j-1); labest; labels2(j:end)];
    end
end

y1 = z(2:end)-z(1:end-1);
y2 = z2(2:end)-z2(1:end-1);

delstr1 = find(y1<prctile(y,25)-IQR*1.5);
delstr2 = find(y2<prctile(y2,25)-IQR*1.5);

labels1([delstr1; delstr2]) = [];
labels2([delstr1; delstr2]) = [];

z([delstr1; delstr2]) = [];
z2([delstr1; delstr2]) = [];

data_in = fin_mat3(:,[2:8,10:16]);

data_in2 = data_in;
data_in2(:,7) = zeros(size(data_in2(:,7)));
data_in2(:,end) = zeros(size(data_in2(:,7)));

data_in2(z,7) = ones(size(z));
data_in2(z2,end) = ones(size(z2));

y = z(2:end) - z(1:end-1);
y2 = z2(2:end) - z2(1:end-1);

stPF = round(z(1:end-1) +.3.*y -15); edPF = round(z(1:end-1) + .6*y);
stPF2 = round(z2(1:end-1) +.3*y2 -15); edPF2 = round(z2(1:end-1) + .6*y2);

strides_to_delete1 = [find(y>prctile(y,75)+IQR*2); find(y<prctile(y,25)-IQR*1.5)];
strides_to_delete2 = [find(y2>prctile(y2,75)+IQR*2); find(y2<prctile(y2,25)-IQR*1.5)];

strides_to_delete1 = [strides_to_delete1; find(labels1==0)-1];
strides_to_delete2 = [strides_to_delete2; find(labels2==0)-1];

data_in = data_in2;
% Calculate angels from data???

data_aug = updownsampled(data_in);
data_aug(data_aug(:,7)~=0,7) = 1;
data_aug(data_aug(:,14)~=0,7) = 1;

dataJitter = Jitter(data_in, .05);
dataScaled = Scaling(data_in,.1);
dataWarped = MagWarp(data_in,.1,100);
dataDist = timeWarp(data_in, .1,100);

figure; 
plot(data_in(:,1))
hold on;
plot(dataJitter(:,1))
plot(dataScaled(:,1))
plot(dataWarped(:,1))
plot(dataDist(:,1))
plot(data_in(:,7),'r')
plot(dataDist(:,7),'--r')
plot(data_aug(:,1))
legend('Orig','Jitter','Scaled','Warped','Time Warped','Strides1','Strides TimeDist','DataAug')

function [xNoise] = Jitter(x,sigma)
noise = normrnd(0,sigma,size(x));
noise(:,[7,14]) = 0;
xNoise = noise + x;
end

function [xScaled] = Scaling(x,sigma)
scalingFactor = normrnd(1, sigma, 1); % Fx1; x is NxF
scalingFactor = repmat(scalingFactor, [size(x,2),1]);
scalingFactor([7 14]) = 1;
xScaled = x.*scalingFactor';
end

function [xWarped] = MagWarp(x,sigma,knots)
xx = (ones([size(x,2),1])*(linspace(0,size(x,1),knots+2)))';
yy = normrnd(1,sigma,1);
yy=repmat(yy,[knots+2,size(x,2)]);
cs = [];
for i = 1:size(x,2)
    cs(:,i) = spline(xx(:,i),yy(:,i),1:size(x,1));
end
cs(:,[7,14]) = 1;
xWarped = x.*cs;
end

function [xTimeDistorted] = timeWarp(x,sig,knots)
xx = (ones([size(x,2),1])*(linspace(0,size(x,1),knots+2)))';
yy = normrnd(1,sig,[knots+2,1]);
% yy=repmat(yy,[1,size(x,2)]);
cs = [];
for i = 1:size(x,2)
    cs(:,i) = spline(xx(:,i),yy,1:size(x,1));
end
tt = cumsum(cs,1);
t_scale = (size(x,1))./tt(end,:);
tt = tt.*t_scale;
xTimeDistorted = [];
for j = 1:size(x,2)
    xTimeDistorted(:,j) = interp1(1:size(x,1),x(:,j),tt(:,j));
end
xTimeDistorted(:,[7,14]) = round(xTimeDistorted(:,[7,14]));
end

function rotation(x)
axis = 2*rand([size(x,2),1]) - 1;
angle = 2*pi*rand(1) - pi;
end

function datamult = updownsampled(x)
x10 = [];
for j = 1:size(x,2)
    x10(:,j) = interp(x(:,j), 10);
end
datamult = [];

for i = 1:10
    datamult = [datamult; x10(i:10:end,:)];
end
end