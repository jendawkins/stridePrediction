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
    InitStance = find(data(:,foot*7)==1);
    cycle_time = InitStance(2:end) - InitStance(1:end-1);
    
%     InitSwing = find(data(:,foot*7)==3);
    InitPF = InitStance(1:end-1) + round(.3*cycle_time);
    InitSwing = InitStance(1:end-1) + round(.6*cycle_time);
    if foot == 1
        Data = data(:,[1,2,3,4,5,6]);
    else
        Data = data(:,[8,9,10,11,12,13]);
    end
    % wherever stance is, define other parts from that;
    if size(data,1)<400
        cycle_time = size(data,1);
        InitSwing = InitStance - .4*cycle_time;
        InitPF = size(data,1);
        
        strideList(1).globalInitStanceSample= InitStance;
        strideList(1).globalInitSwingSample = InitSwing;
        for nm = 1:length(raw_sensor_outputs)
            strideList(1).(raw_sensor_outputs{nm}) = Data(:,nm);
        end
    end
    
%     names = {'aAccX','aAccY','aAccZ','gVelX','gVelY','gVelZ'};
    for nm = 1:length(raw_sensor_outputs)
        for k = 1:length(InitStance)-1
            % foot = 1 : 1:6; foot = 2: 8:13
            try
                strideList(k).globalInitStanceSample = InitStance(k);
            end
            try
                strideList(k).globalInitSwingSample = InitSwing(k);
            end
            try
                strideList(k).(raw_sensor_outputs{nm}) = Data(InitStance(k):InitStance(k+1)-1,nm);
%                 strideList(k).(raw_sensor_outputs{nm}) = Data(InitStance(k):InitStance(k+1),nm);
            end
        end
    end
    lenStrideList = length(strideList);
    
%     disp('Calculating integrals and derivatives...');
    
    scale_factors = {ACCEL_LSB_PER_MPS2, -1.0*ACCEL_LSB_PER_MPS2, ACCEL_LSB_PER_MPS2, ...
        GYRO_LSB_PER_RAD, GYRO_LSB_PER_RAD, GYRO_LSB_PER_RAD};
    % %derivative_outputs = {'d1aAccX','d1aAccY','d1aAccZ','d2aAccX','d2aAccY','d2aAccZ','d1aOmegaX','d1aOmegaY','d1aOmegaZ','d2aOmegaX','d2aOmegaY','d2aOmegaZ'};
    % %integral_outputs = {'i1aAccX','i1aAccY','i1aAccZ','i2aAccX','i2aAccY','i2aAccZ','i1aOmegaX','i1aOmegaY','i1aOmegaZ','i2aOmegaX','i2aOmegaY','i2aOmegaZ'};
    
    for i=1:lenStrideList
        strideSamples = numel(strideList(i).a1Raw);
        
        %Scale accelerometer and gyroscope outputs
        for j=1:6
            strideList(i).(scaled_sensor_outputs{j}) = ...
                strideList(i).(raw_sensor_outputs{j}) / scale_factors{j};
        end
        
        %Initialize integrals and derivatives
        for k=1:numel(derivative_outputs)
            strideList(i).(derivative_outputs{k}) = zeros(strideSamples,1);
            strideList(i).(integral_outputs{k}) = zeros(strideSamples,1);
        end
        
        %Calculate integrals and derivatives
        for j=2:strideSamples
            for k=1:numel(derivative_outputs)
                der_field = derivative_outputs{k};
                int_field = integral_outputs{k};
                val_field = der_field(3:end);
                strideList(i).(der_field)(j) =  FILTB*(strideList(i).(val_field)(j) - strideList(i).(val_field)(j-1))+FILTA*strideList(i).(der_field)(j-1);
                strideList(i).(int_field)(j) =  strideList(i).(int_field)(j-1) + strideList(i).(val_field)(j);
            end
        end
    end
    
    
    
%     disp('Calculating a priori inverse kinematics...');
    
    %Go through all strides individually
    for i=1:lenStrideList
        strideSamples = numel(strideList(i).aAccX);
%         stanceSamples = strideSamples;
        stanceSamples = strideList(i).globalInitSwingSample - strideList(i).globalInitStanceSample;
        
        %Add inverse kinematics outputs
        for j=1:numel(inverse_kinematics_outputs)
            strideList(i).(inverse_kinematics_outputs{j}) = zeros(strideSamples,1);
        end
        
        
        %Find latest foot static time
%         tqDot = zeros(strideSamples,1);
        accNormSq = zeros(strideSamples,1);
        foundFirstFootStatic = 0;
        latestFootStaticSample = 2;
        resetTrigger = 0;
        for j=2:stanceSamples
%             tqDot(j) = 0.2*tqDot(j-1) + 0.8*(strideList(i).torqueRaw(j) - strideList(i).torqueRaw(j-1));
            accNormSq(j) = sumsqr([strideList(i).aAccY(j) strideList(i).aAccZ(j)]);
            
            %Necessary condition
%             if (abs(strideList(i).torqueRaw(j)) > MIN_TQ_FOR_FOOT_STATIC_NM)
                %                 trial.biom.tqDot(i) > constants.MIN_TQDOT_FOR_FOOT_STATIC && ...
                %                 trial.biom.tq(i) > trial.flags.maxTq - 20)
                
                %Default case
                if (j - latestFootStaticSample > DEFAULT_ZVUP_SAMPLES && ~foundFirstFootStatic)
                    latestFootStaticSample = j;
                    resetTrigger = 1;
                end
                
                %Optimal case
                if (accNormSq(j) < UPPER_ACCNORM_THRESH_SQ && ...
                        accNormSq(j) > LOWER_ACCNORM_THRESH_SQ)
                    foundFirstFootStatic = 1;
                    latestFootStaticSample = j;
                    resetTrigger = 2;
                end
%             end
        end
        
        strideList(i).globalFootStaticSample = strideList(i).globalInitStanceSample + latestFootStaticSample - 1;
        strideList(i).resetTrigger = resetTrigger;
        
        %Set initial attitude matrix R
        zAccWithCentripetalAccCompensation = strideList(i).aAccZ(latestFootStaticSample) + strideList(i).aOmegaX(latestFootStaticSample)^2*ANKLE_TO_IMU_SAGITTAL_PLANE_M;
        yAccWithTangentialAccCompensation = strideList(i).aAccY(latestFootStaticSample) + strideList(i).d1aOmegaX(latestFootStaticSample)*SAMPLE_RATE_HZ*ANKLE_TO_IMU_SAGITTAL_PLANE_M;
        accNorm = norm([yAccWithTangentialAccCompensation zAccWithCentripetalAccCompensation]);
        
        %     zAccWithCentripetalAccCompensation = strideList(i).aAccZ(latestFootStaticSample) ;
        %     yAccWithTangentialAccCompensation = strideList(i).aAccY(latestFootStaticSample);
        %     accNorm = norm([yAccWithTangentialAccCompensation zAccWithCentripetalAccCompensation]);
        
        cospitch = zAccWithCentripetalAccCompensation/accNorm;
        sinpitch = yAccWithTangentialAccCompensation/accNorm;
        R = [cospitch -sinpitch; sinpitch cospitch];
        strideList(i).r0(latestFootStaticSample) = R(1,1);
        strideList(i).r1(latestFootStaticSample) = R(1,2);
        strideList(i).r2(latestFootStaticSample) = R(2,1);
        strideList(i).r3(latestFootStaticSample) = R(2,2);
        
        %Set initial IMU position and velocity in global frame
        strideList(i).gVelY(latestFootStaticSample) = 0.0;
        strideList(i).gVelZ(latestFootStaticSample) = 0.0;
        strideList(i).gPosY(latestFootStaticSample) = 0.0;
        strideList(i).gPosZ(latestFootStaticSample) = 0.0;
        
        %Reset integrals
        for k=1:numel(integral_outputs)
            strideList(i).(integral_outputs{k}) = strideList(i).(integral_outputs{k}) - strideList(i).(integral_outputs{k})(latestFootStaticSample);
        end
        
        %Calculate inverse kinematics
        for j=latestFootStaticSample:strideSamples
            strideList(i).pitch(j) = acos(R(1,1));
            Om = [0 -1*strideList(i).aOmegaX(j); strideList(i).aOmegaX(j) 0];
            R = R*(eye(2) + Om * SAMPLE_PERIOD_S);
            
            strideList(i).r0(j) = R(1,1);
            strideList(i).r1(j) = R(1,2);
            strideList(i).r2(j) = R(2,1);
            strideList(i).r3(j) = R(2,2);
            
            strideList(i).gAccY(j) = R(1,:)*[strideList(i).aAccY(j); strideList(i).aAccZ(j)];
            strideList(i).gAccZ(j) = R(2,:)*[strideList(i).aAccY(j); strideList(i).aAccZ(j)] - GRAVITY_MPS2;
            strideList(i).gVelY(j) = strideList(i).gVelY(j-1)+SAMPLE_PERIOD_S*strideList(i).gAccY(j);
            strideList(i).gVelZ(j) = strideList(i).gVelZ(j-1)+SAMPLE_PERIOD_S*strideList(i).gAccZ(j);
            strideList(i).gPosY(j) = strideList(i).gPosY(j-1)+SAMPLE_PERIOD_S*strideList(i).gVelY(j);
            strideList(i).gPosZ(j) = strideList(i).gPosZ(j-1)+SAMPLE_PERIOD_S*strideList(i).gVelZ(j);
        end
        
    end
    FootStr.(['F' num2str(foot)]) = strideList;
end
end