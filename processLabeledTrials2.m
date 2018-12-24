%% processedLabeledTrials script
% This function runs a simulation on data collected from Matt Carney's TF8
% ankle-foot prosthesis. In particular, it simulates using data from the
% onboard ankle angle encoder, torque sensor, and inertial measurement unit
% to predict among five different terrains for every stride taken by the
% subject. Once the stride is taken, there is a back-estimation step, which
% allows for very accurate labeling of the stride after it has been taken,
% and a subsequent update of the machine learning model based on this new
% information. The trials are contained in the struct RC053018 and have
% been manually labeled using knowledge of the walking terrain from video.
%%


function strideList = processLabeledTrials(RC053018)

%% Define necessary constants

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

%% Define signal lists
raw_sensor_outputs = {'a1Raw','a2Raw','a3Raw','g1Raw','g2Raw','g3Raw'};
scaled_sensor_outputs = {'aAccX','aAccZ','aAccY','aOmegaX','aOmegaZ','aOmegaY'};
derivative_outputs = {'d1aAccX','d1aAccY','d1aAccZ','d1aOmegaX','d1aOmegaY','d1aOmegaZ'};
integral_outputs = {'i1aAccX','i1aAccY','i1aAccZ','i1aOmegaX','i1aOmegaY','i1aOmegaZ'};
inverse_kinematics_outputs = {'r0','r1','r2','r3','gAccY','gAccZ','gVelY','gVelZ','gPosY','gPosZ'};
prediction_signals = ['aAccY','aAccZ','aOmegaX','d1aAccY','d1aAccZ','d1aOmegaX','i1aAccY','i1aAccZ','i1aOmegaX', inverse_kinematics_outputs];


if (nargin == 1)
%% Divide data into individual strides, making a stride list
disp('Dividing data into individual strides...');
strideList = [];
trials = fieldnames(RC053018);
dataFields = fieldnames(RC053018.(trials{1}));
for h=1:numel(trials)
    currentTrial = RC053018.(trials{h});
    
    highTorqueInds = abs(currentTrial.torqueRaw) > GAIT_EVENT_THRESHOLD_TORQUE_NM;
    
    %Patch short swing periods with stance (so default is stance unless
    %you've been in swing for greater than MIN_SWING_SAMPLES)
    i = 1;
    while (i < numel(highTorqueInds)-MIN_SWING_SAMPLES)
        if (find(highTorqueInds(i:i+MIN_SWING_SAMPLES)))
            highTorqueInds(i:i+MIN_SWING_SAMPLES) = deal(1);
            i = i+MIN_SWING_SAMPLES;
        end
        i = i + 1;
    end
    
    %Find gait event inds
    gaitEventInds = diff(highTorqueInds);
    stanceStartInds = find(gaitEventInds == 1);
    swingStartInds = find(gaitEventInds == -1);
    firstStanceInd = stanceStartInds(1);
    swingStartInds(swingStartInds < firstStanceInd) = [];
    
    %Compile list of stride structs
    for i=1:numel(stanceStartInds)-2
        stride.globalInitStanceSample = stanceStartInds(i);
        stride.globalInitSwingSample = swingStartInds(i);
        stride.globalTargStanceSample = stanceStartInds(i+1);
        stride.globalTargSwingSample = swingStartInds(i+1);
        stride.trialName = trials{h};
        for k=1:numel(dataFields)
            stride.(dataFields{k}) = currentTrial.(dataFields{k})(stanceStartInds(i):swingStartInds(i+1));
        end
        stride.manualLabel = stride.manualLabels(stride.globalTargStanceSample - stride.globalInitStanceSample);
        strideList = [strideList; stride];
    end 
   
end

%% Filter stride list
disp('Filtering stride list...');
stridesMarkedForDeletion = find([strideList.globalTargStanceSample] - [strideList.globalInitStanceSample] < MIN_STRIDE_SAMPLES | ...
    [strideList.globalTargStanceSample] - [strideList.globalInitSwingSample] < POST_SWING_CUTOFF_SAMPLES);
strideList(stridesMarkedForDeletion) = [];
lenStrideList = numel(strideList);

%% Calculate integrals and derivatives of accel and gyro values
disp('Calculating integrals and derivatives...');

scale_factors = {ACCEL_LSB_PER_MPS2, -1.0*ACCEL_LSB_PER_MPS2, ACCEL_LSB_PER_MPS2, ...
    GYRO_LSB_PER_RAD, GYRO_LSB_PER_RAD, GYRO_LSB_PER_RAD};
% %derivative_outputs = {'d1aAccX','d1aAccY','d1aAccZ','d2aAccX','d2aAccY','d2aAccZ','d1aOmegaX','d1aOmegaY','d1aOmegaZ','d2aOmegaX','d2aOmegaY','d2aOmegaZ'};
% %integral_outputs = {'i1aAccX','i1aAccY','i1aAccZ','i2aAccX','i2aAccY','i2aAccZ','i1aOmegaX','i1aOmegaY','i1aOmegaZ','i2aOmegaX','i2aOmegaY','i2aOmegaZ'};

for i=1:lenStrideList
    strideSamples = numel(strideList(i).torqueRaw);
    
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



%% Calculate a priori inverse kinematics for each stride (for prediction)
disp('Calculating a priori inverse kinematics...');

%Go through all strides individually
 for i=1:lenStrideList
     strideSamples = numel(strideList(i).torqueRaw);
     stanceSamples = strideList(i).globalInitSwingSample - strideList(i).globalInitStanceSample;
     
     %Add inverse kinematics outputs
     for j=1:numel(inverse_kinematics_outputs)
         strideList(i).(inverse_kinematics_outputs{j}) = zeros(strideSamples,1);
     end
    
        
    %Find latest foot static time
    tqDot = zeros(strideSamples,1);
    accNormSq = zeros(strideSamples,1);
    foundFirstFootStatic = 0;
    latestFootStaticSample = 2;
    resetTrigger = 0;
    for j=2:stanceSamples
        tqDot(j) = 0.2*tqDot(j-1) + 0.8*(strideList(i).torqueRaw(j) - strideList(i).torqueRaw(j-1));
        accNormSq(j) = sumsqr([strideList(i).aAccY(j) strideList(i).aAccZ(j)]);
        
        %Necessary condition
        if (abs(strideList(i).torqueRaw(j)) > MIN_TQ_FOR_FOOT_STATIC_NM)
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
        end
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


%% Do some analytics on a priori inverse kinematics 
disp('Doing some analytics...');
s4 = strideList([strideList.manualLabel] == 4);
s5 = strideList([strideList.manualLabel] == 5);
zfinal4 = zeros(numel(s4),1);
zfinal5 = zeros(numel(s5),1);
for i=1:numel(s4)
    zfinal4(i) = s4(i).gPosZ(end);
end
for i=1:numel(s5)
    zfinal5(i) = s5(i).gPosZ(end);
end
disp (['Mean/var of Ustairs final Z: ',num2str(mean(zfinal4)), ' / ', num2str(var(zfinal4))]);
disp (['Mean/var of Dstairs final Z: ',num2str(mean(zfinal5)), ' / ', num2str(var(zfinal5))]);

else
    lenStrideList = numel(strideList);
end

%% Extract predictor features
disp('Extracting predictor features...');
nOutputSignals = numel(prediction_signals);

means = zeros(lenStrideList, numel(prediction_signals));
maxs = means;
mins = means;
ranges = means;

for i=1:lenStrideList
    window_start = strideList(i).globalFootStaticSample - strideList(i).globalInitStanceSample + 1;
    window_end = strideList(i).globalInitSwingSample - strideList(i).globalInitStanceSample + POST_SWING_CUTOFF_SAMPLES;
    for j=1:nOutputSignals
        means(i,j) = mean(strideList(i).(prediction_signals{j})(window_start:window_end));
        maxs(i,j) = max(strideList(i).(prediction_signals{j})(window_start:window_end));
        mins(i,j) = min(strideList(i).(prediction_signals{j})(window_start:window_end));
        ranges(i,j) = range(strideList(i).(prediction_signals{j})(window_start:window_end));      
    end
end

feature_matrix = [maxs, mins, ranges];
nOutputFeatures = nOutputSignals * 3;

%% Back estimate terrain label
disp('Back estimating terrain label...');
estimated_labels = backEstimate(strideList);


% save('simulation_data','strideList','feature_matrix','estimated_labels');
%% Run full algorithm many times over
disp('Running prediction / back estimation simulation...');
predicted_labels_matrix = zeros(lenStrideList,N_SIMULATIONS);
estimated_labels_matrix = zeros(lenStrideList,N_SIMULATIONS);

for h=1:N_SIMULATIONS

    %Randomize the order of strides
    rperm = randperm(lenStrideList);
    randomized_feature_matrix = feature_matrix(rperm,:);
    estimated_labels_matrix(:,h) = estimated_labels(rperm);
    
    % Initialize statistical variables
    prior = [1,1,1,1,1];
    intraclass_mu = zeros(N_CLASSES,nOutputFeatures);
    interclass_mu = zeros(1,nOutputFeatures);
    interclass_sigma = eye(nOutputFeatures);

    %Initialize LDA predictor parameters
    A = cell(N_CLASSES,1);
    B = cell(N_CLASSES,1);
    for i=1:N_CLASSES
        A{i} = zeros(1,nOutputFeatures);
        B{i} = 0;
    end

%     %Specify whether using manual labels or estimator labels
%     usingManualLabels = 0;
%     if (usingManualLabels)
%         labels = [strideList.manualLabel];
%     else
%         labels = estimatedLabels;
%     end


    discriminant_function_outputs = zeros(N_CLASSES,1);
    class_populations = zeros(N_CLASSES,1);
    for i=1:lenStrideList
        
        new_stride = randomized_feature_matrix(i,:);
        label = estimated_labels_matrix(i,h);
        
        %calculate new discriminant function parameters and make prediction

        A{label} = intraclass_mu(label,:) * inv(interclass_sigma/i);
        B{label} = -0.5 * A{label} * intraclass_mu(label,:)' + log(prior(label));

        for j=1:N_CLASSES
            discriminant_function_outputs(j) = A{j} * new_stride' + B{j};
        end
        [~,predicted_labels_matrix(i,h)] = max(discriminant_function_outputs);

        %update class means using back estimated label
        class_populations(label) = class_populations(label) + 1;
        intraclass_mu(label,:) = 1.0/class_populations(label) * ((class_populations(label)-1)*intraclass_mu(label,:) + new_stride);

        %update global mean using back estimated label
        old_interclass_mu = interclass_mu;
        interclass_mu = 1/i * ((i-1)*interclass_mu + new_stride);

        %update feature covariance with welford's algorithm using back
        %estimated label
        features_offset_by_old_mean = new_stride - old_interclass_mu;
        features_offset_by_mean = new_stride - interclass_mu;
        interclass_sigma = interclass_sigma + features_offset_by_mean'*features_offset_by_old_mean;
    end
    
end

disp('end')

end
