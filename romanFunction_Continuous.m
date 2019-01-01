function [FootStr] = romanFunction(data)
%Physical constants
GRAVITY_MPS2 = 9.8;
RAD_PER_DEG = pi/180;

%System constants
GYRO_LSB_PER_DPS = 32.8; %per http://dephy.com/wiki/flexsea/doku.php?id=units
ACCEL_LSB_PER_G = 8192; %per http://dephy.com/wiki/flexsea/doku.php?id=units
SAMPLE_RATE_HZ = 100;
ANKLE_POS_IMU_FRAME_X_M = 0.0; %Frontal axis (medial->lateral)
ANKLE_POS_IMU_FRAME_Y_M = -0.00445; %Longitudinal axis (bottom->top)
ANKLE_POS_IMU_FRAME_Z_M = -0.0605; %Sagittal axis (back->front)

%Machine learning
POST_SWING_CUTOFF_TIME_S = 0.2;
N_CLASSES = 5;
N_SIMULATIONS = 10;

%Gait event detection
GAIT_EVENT_THRESHOLD_TORQUE_NM = 1;
MIN_TQDOT_FOR_FOOT_STATIC_NM = -0.1;
MIN_TQ_FOR_FOOT_STATIC_NM = 15;
DEFAULT_ZVUP_TIME_S = 0.05;
UPPER_ACCNORM_THRESH_SQ = 102.01;
LOWER_ACCNORM_THRESH_SQ = 90.25;
MIN_SWING_TIME_S = 0.1; % was 0.32;
MIN_STANCE_TIME_S = 0.1;
MIN_STRIDE_TIME_S = 0.5;


%Default filter coefficients
FILTA = 0.95;
FILTB = 0.05;

%% Define derived constants
MIN_SWING_SAMPLES = MIN_SWING_TIME_S * SAMPLE_RATE_HZ;
MIN_STANCE_SAMPLES = MIN_STANCE_TIME_S * SAMPLE_RATE_HZ;
MIN_STRIDE_SAMPLES = MIN_STRIDE_TIME_S * SAMPLE_RATE_HZ;
ACCEL_LSB_PER_MPS2 = ACCEL_LSB_PER_G / GRAVITY_MPS2;
GYRO_LSB_PER_RAD = GYRO_LSB_PER_DPS / RAD_PER_DEG;
DEFAULT_ZVUP_SAMPLES = DEFAULT_ZVUP_TIME_S * SAMPLE_RATE_HZ;
ANKLE_TO_IMU_SAGITTAL_PLANE_M = hypot(ANKLE_POS_IMU_FRAME_Y_M, ANKLE_POS_IMU_FRAME_Z_M);
SAMPLE_PERIOD_S = 1/SAMPLE_RATE_HZ;
POST_SWING_CUTOFF_SAMPLES = int32(POST_SWING_CUTOFF_TIME_S * SAMPLE_RATE_HZ);

raw_sensor_outputs = {'a1Raw','a2Raw','a3Raw','g1Raw','g2Raw','g3Raw'};
scaled_sensor_outputs = {'aAccX','aAccZ','aAccY','aOmegaX','aOmegaZ','aOmegaY'};
derivative_outputs = {'d1aAccX','d1aAccY','d1aAccZ','d1aOmegaX','d1aOmegaY','d1aOmegaZ'};
integral_outputs = {'i1aAccX','i1aAccY','i1aAccZ','i1aOmegaX','i1aOmegaY','i1aOmegaZ'};
inverse_kinematics_outputs = {'r0','r1','r2','r3','gAccY','gAccZ','gVelY','gVelZ','gPosY','gPosZ'};
prediction_signals = ['aAccY','aAccZ','aOmegaX','d1aAccY','d1aAccZ','d1aOmegaX','i1aAccY','i1aAccZ','i1aOmegaX', inverse_kinematics_outputs];

for foot = 1:2
%     InitStance = find(data(:,foot*7)==1);
%     if foot == 1
%         start_data = InitStance;
%     end
%     InitStance(InitStance<start_data(1)) = [];
%     if length(start_data)>length(InitStance)
%         InitStance = [InitStance; length(data)];
%     end
%     cycle_time = start_data(2:end) - start_data(1:end-1);
    
    %     InitSwing = find(data(:,foot*7)==3);
%     InitPF = InitStance(1:end-1) + round(.3*cycle_time);
%     InitSwing = InitStance(1:end-1) - round(.4*cycle_time);
    if foot == 1
        Data = data(:,[1,2,3,4,5,6]);
    else
        Data = data(:,[8,9,10,11,12,13]);
    end
    % wherever stance is, define other parts from that;
    if size(data,1)<400
%         cycle_time = size(data,1);
%         InitSwing = InitStance - .4*cycle_time;
%         InitPF = size(data,1);
%         
%         strideList(1).globalInitStanceSample= InitStance;
%         strideList(1).globalInitSwingSample = InitSwing;
        for nm = 1:length(raw_sensor_outputs)
            strideList.(raw_sensor_outputs{nm}) = Data(:,nm);
        end
%         strideList(1).HSmag = sqrt((Data(InitStance,1).^2 + Data(InitStance,2).^2)...
%                     +Data(InitStance,3).^2);
    else
        
        %     names = {'aAccX','aAccY','aAccZ','gVelX','gVelY','gVelZ'};
        for nm = 1:length(raw_sensor_outputs)
                % foot = 1 : 1:6; foot = 2: 8:13
%                     strideList.globalInitStanceSample = InitStance;
%                     strideList.HSmag = sqrt((Data(InitStance,1).^2 + Data(InitStance,2).^2)...
%                         +Data(InitStance,3).^2);
%                     strideList.globalInitSwingSample = InitSwing;
%                     strideList(k).globalInitSwingSample = InitSwing(k);
                    strideList.(raw_sensor_outputs{nm}) = Data(:,nm);
                    %                 strideList(k).(raw_sensor_outputs{nm}) = Data(InitStance(k):InitStance(k+1),nm);

        end
    end
%     lenStrideList = length(strideList);
%     rawsiglen = 0;
%     cumulen = [];
%     heel_strike = [];
%     swing = [];
%     for kk = 1:lenStrideList
%         rawsiglen = rawsiglen + length(strideList(kk).a1Raw);
%         cumulen = [cumulen; rawsiglen];
%         heel_strike(kk) = strideList(kk).globalInitStanceSample;
%         swing(kk) = strideList(kk).globalInitSwingSample;
%     end
%     disp('Calculating integrals and derivatives...');
    
    scale_factors = {ACCEL_LSB_PER_MPS2, -1.0*ACCEL_LSB_PER_MPS2, ACCEL_LSB_PER_MPS2, ...
        GYRO_LSB_PER_RAD, GYRO_LSB_PER_RAD, GYRO_LSB_PER_RAD};
    % %derivative_outputs = {'d1aAccX','d1aAccY','d1aAccZ','d2aAccX','d2aAccY','d2aAccZ','d1aOmegaX','d1aOmegaY','d1aOmegaZ','d2aOmegaX','d2aOmegaY','d2aOmegaZ'};
    % %integral_outputs = {'i1aAccX','i1aAccY','i1aAccZ','i2aAccX','i2aAccY','i2aAccZ','i1aOmegaX','i1aOmegaY','i1aOmegaZ','i2aOmegaX','i2aOmegaY','i2aOmegaZ'};
    
        strideSamples = numel(strideList.a1Raw);
        
        %Scale accelerometer and gyroscope outputs
        for j=1:6
            strideList.(scaled_sensor_outputs{j}) = ...
                strideList.(raw_sensor_outputs{j}) / scale_factors{j};
        end
               
        %Initialize integrals and derivatives
        for k=1:numel(derivative_outputs)
            strideList.(derivative_outputs{k}) = zeros(strideSamples,1);
            strideList.(integral_outputs{k}) = zeros(strideSamples,1);
        end
        
        %Calculate integrals and derivatives
        for j=2:strideSamples
            for k=1:numel(derivative_outputs)
                der_field = derivative_outputs{k};
                int_field = integral_outputs{k};
                val_field = der_field(3:end);
                strideList.(der_field)(j) =  FILTB*(strideList.(val_field)(j) - strideList.(val_field)(j-1))+FILTA*strideList.(der_field)(j-1);
                strideList.(int_field)(j) =  strideList.(int_field)(j-1) + strideList.(val_field)(j);
            end
        end
    
    
    
%     disp('Calculating a priori inverse kinematics...');
    
    %Go through all strides individually
FootStr.(['F' num2str(foot)]) = strideList;
end
end
