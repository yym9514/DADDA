%% Inverse Generator (IG) %%
clc; clear all; close all; warning off;


%% Load input file
% Design variables
load("designVariable_target");

% Response
load("response1_target");

for i = 1:size(response_target,2)/2
    response_4D(:,:,1,i) = [response_target(:,2*i)];
end

augimds = augmentedImageDatastore([size(response_target,1) 1],response_4D);


%% Network structure
% Inverse generator (IG)
filterSize = [34 1];

layersIG = [
    
    imageInputLayer([size(response_target,1) 1 1],'Normalization','none','Name','in')
 
    convolution2dLayer(filterSize,8,'Stride',2,'Padding',0,'Name','conv1')
    batchNormalizationLayer('Name','batch1')
    reluLayer('Name','relu1')
        
    convolution2dLayer(filterSize,16,'Stride',2,'Padding',0,'Name','conv2')
    batchNormalizationLayer('Name','batch2')
    reluLayer('Name','relu2')
        
    convolution2dLayer(filterSize,32,'Stride',2,'Padding',0,'Name','conv3')
    batchNormalizationLayer('Name','batch3')
    reluLayer('Name','relu3')
        
    convolution2dLayer(filterSize,64,'Stride',2,'Padding',0,'Name','conv4')
    batchNormalizationLayer('Name','batch4')
    reluLayer('Name','relu4')
        
    convolution2dLayer(filterSize,5,'Stride',2,'Padding',0,'Name','conv5')
    batchNormalizationLayer('Name','batch5')
    reluLayer('Name','relu5')
    
    fullyConnectedLayer(5,'Name','fc') 
    regressionLayer('Name','reg')
    
    ];


%% Specify Training Options
epoch = 50000;
miniBatchSize = 20;
learnRateIG = 0.001;


%% Train Model
x1_training = designVariable_target(:,1);
x2_training = designVariable_target(:,2);
x3_training = designVariable_target(:,3);
x4_training = designVariable_target(:,4);
x5_training = designVariable_target(:,5);

for i = 1:size(designVariable_target,1)
    x_training(i,1) = [x1_training(i,1)];
    x_training(i,2) = [x2_training(i,1)];
    x_training(i,3) = [x3_training(i,1)];
    x_training(i,4) = [x4_training(i,1)];
    x_training(i,5) = [x5_training(i,1)];
end

x_train = x_training;
x_validation = x_training;
response_y = read(augimds);
y_training = cat(4,response_y{:,1}{:});
y_train = y_training;
y_validation = y_training;

validationFrequency = floor(numel(x_train)/miniBatchSize);

options = trainingOptions('adam', ...
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',epoch, ...
    'InitialLearnRate',learnRateIG, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',learnRateIG, ...
    'LearnRateDropPeriod',epoch, ...
    'Shuffle','never', ...
    'ValidationData',{y_validation,x_validation}, ...
    'ValidationFrequency',validationFrequency, ...
    'Plots','training-progress', ...
    'Verbose',false);

[dlnetIG1,info] = trainNetwork(y_train,x_train,layersIG,options);
x_prediction = predict(dlnetIG1,y_validation);

predictionError = x_validation - x_prediction;
squares = predictionError.^2;
rmse = sqrt(mean(squares));