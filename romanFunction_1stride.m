function [strideList] = romanFunction_1stride(stride)
%Physical constants
GRAVITY_MPS2 = 9.8;
RAD_PER_DEG = pi/180;

%System constants
GYRO_LSB_PER_DPS = 32.8; %per http://dephy.com/wiki/flexsea/doku.php?id=units
ACCEL_LSB_PER_G = 8192; %per http://dephy.com/wiki/flexsea/doku.php?id=units
SAMPLE_RATE_HZ = 1000;
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

Data = stride;
%     names = {'aAccX','aAccY','aAccZ','gVelX','gVelY','gVelZ'};
for nm = 1:length(raw_sensor_outputs)
    % foot = 1 : 1:6; foot = 2: 8:13
    strideList.(raw_sensor_outputs{nm}) = Data(:,nm);
end
% strideList.globalInitSwingSample = find(Data(:,7)==3);
% strideList.globalInitStanceSample = find(Data(:,7)==1);

% disp('Calculating integrals and derivatives...');

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




% disp('Calculating a priori inverse kinematics...');

%Go through all strides individually
    strideSamples = numel(strideList.aAccX);
%     stanceSamples = strideList.globalInitSwingSample - strideList.globalInitStanceSample;
%     
    %Add inverse kinematics outputs
    for j=1:numel(inverse_kinematics_outputs)
        strideList.(inverse_kinematics_outputs{j}) = zeros(strideSamples,1);
    end
    
    
    %Find latest foot static time
    %         tqDot = zeros(strideSamples,1);
    accNormSq = zeros(strideSamples,1);
    foundFirstFootStatic = 0;
    latestFootStaticSample = 2;
    resetTrigger = 0;
    for j=2:strideSamples
        %             tqDot(j) = 0.2*tqDot(j-1) + 0.8*(strideList(i).torqueRaw(j) - strideList(i).torqueRaw(j-1));
        accNormSq(j) = sumsqr([strideList.aAccY(j) strideList.aAccZ(j)]);
        
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
    
%     strideList.globalFootStaticSample = strideList.globalInitStanceSample + latestFootStaticSample - 1;
    strideList.resetTrigger = resetTrigger;
    
    %Set initial attitude matrix R
    zAccWithCentripetalAccCompensation = strideList.aAccZ(latestFootStaticSample) + strideList.aOmegaX(latestFootStaticSample)^2*ANKLE_TO_IMU_SAGITTAL_PLANE_M;
    yAccWithTangentialAccCompensation = strideList.aAccY(latestFootStaticSample) + strideList.d1aOmegaX(latestFootStaticSample)*SAMPLE_RATE_HZ*ANKLE_TO_IMU_SAGITTAL_PLANE_M;
    accNorm = norm([yAccWithTangentialAccCompensation zAccWithCentripetalAccCompensation]);
    
    %     zAccWithCentripetalAccCompensation = strideList(i).aAccZ(latestFootStaticSample) ;
    %     yAccWithTangentialAccCompensation = strideList(i).aAccY(latestFootStaticSample);
    %     accNorm = norm([yAccWithTangentialAccCompensation zAccWithCentripetalAccCompensation]);
    
    cospitch = zAccWithCentripetalAccCompensation/accNorm;
    sinpitch = yAccWithTangentialAccCompensation/accNorm;
    R = [cospitch -sinpitch; sinpitch cospitch];
    strideList.r0(latestFootStaticSample) = R(1,1);
    strideList.r1(latestFootStaticSample) = R(1,2);
    strideList.r2(latestFootStaticSample) = R(2,1);
    strideList.r3(latestFootStaticSample) = R(2,2);
    
    %Set initial IMU position and velocity in global frame
    strideList.gVelY(latestFootStaticSample) = 0.0;
    strideList.gVelZ(latestFootStaticSample) = 0.0;
    strideList.gPosY(latestFootStaticSample) = 0.0;
    strideList.gPosZ(latestFootStaticSample) = 0.0;
    
    %Reset integrals
    for k=1:numel(integral_outputs)
        strideList.(integral_outputs{k}) = strideList.(integral_outputs{k}) - strideList.(integral_outputs{k})(latestFootStaticSample);
    end
    
    %Calculate inverse kinematics
    for j=latestFootStaticSample:strideSamples
        strideList.pitch(j) = acos(R(1,1));
        Om = [0 -1*strideList.aOmegaX(j); strideList.aOmegaX(j) 0];
        R = R*(eye(2) + Om * SAMPLE_PERIOD_S);
        
        strideList.r0(j) = R(1,1);
        strideList.r1(j) = R(1,2);
        strideList.r2(j) = R(2,1);
        strideList.r3(j) = R(2,2);
        
        strideList.gAccY(j) = R(1,:)*[strideList.aAccY(j); strideList.aAccZ(j)];
        strideList.gAccZ(j) = R(2,:)*[strideList.aAccY(j); strideList.aAccZ(j)] - GRAVITY_MPS2;
        strideList.gVelY(j) = strideList.gVelY(j-1)+SAMPLE_PERIOD_S*strideList.gAccY(j);
        strideList.gVelZ(j) = strideList.gVelZ(j-1)+SAMPLE_PERIOD_S*strideList.gAccZ(j);
        strideList.gPosY(j) = strideList.gPosY(j-1)+SAMPLE_PERIOD_S*strideList.gVelY(j);
        strideList.gPosZ(j) = strideList.gPosZ(j-1)+SAMPLE_PERIOD_S*strideList.gVelZ(j);
    end
    
end
