%% Designable Variational AutoEncoder (DVAE) %%
clc; clear all; close all; warning off;

global dlnetD1 dlnetD2


%% Load Input File
% Responses
load("response1_source");  % Vibration acceleration
load("response2_source");  % Lateral deviation

for i = 1:size(response1_source,2)/2
    response1_time_source(:,i) = [response1_source(:,2*i-1)];
    response1_value_source(:,i) = [response1_source(:,2*i)];
    response1_4DValue_source(:,:,1,i) = [response1_value_source(:,i)];
end

for i = 1:size(response2_source,2)/2
    response2_time_source(:,i) = [response2_source(:,2*i-1)];
    response2_value_source(:,i) = [response2_source(:,2*i)];
    response2_4DValue_source(:,:,1,i) = [response2_value_source(:,i)];
end

augimds1 = augmentedImageDatastore([size(response1_source,1) 1],response1_4DValue_source);
augimds2 = augmentedImageDatastore([size(response2_source,1) 1],response2_4DValue_source);


%% Network structure (Response1)
% Encoder (E)
filterSize = [34 1];
latentDim = 5;

layersE1 = [
    
    imageInputLayer([size(response1_source,1) 1 1],'Normalization','none','Name','in')
    
    convolution2dLayer(filterSize,8,'Stride',2,'Padding',0,'Name','conv1')
    reluLayer('Name','relu1')
    
    convolution2dLayer(filterSize,16,'Stride',2,'Padding',0,'Name','conv2')
    reluLayer('Name','relu2')
    
    convolution2dLayer(filterSize,32,'Stride',2,'Padding',0,'Name','conv3')
    reluLayer('Name','relu3')
    
    convolution2dLayer(filterSize,64,'Stride',2,'Padding',0,'Name','conv4')
    reluLayer('Name','relu4')
    
    fullyConnectedLayer(2*latentDim,'Name','fc_encoder')
    
    ];

lgraphE1 = layerGraph(layersE1);
dlnetE1 = dlnetwork(lgraphE1);


% Decoder (D)
layersD1 = [
    
    imageInputLayer([1 1 latentDim],'Normalization','none','Name','in')
    
    transposedConv2dLayer(filterSize,64,'Name','tcov1')
    reluLayer('Name','relu1')
    
    transposedConv2dLayer(filterSize,32,'Stride',2,'Cropping',0,'Name','tcov2')
    reluLayer('Name','relu2')
    
    transposedConv2dLayer(filterSize,16,'Stride',2,'Cropping',0,'Name','tcov3')
    reluLayer('Name','relu3')
    
    transposedConv2dLayer(filterSize,8,'Stride',2,'Cropping',0,'Name','tcov4')
    reluLayer('Name','relu4')
    
    transposedConv2dLayer(filterSize,1,'Stride',2,'Cropping',0,'Name','tcov5')
    tanhLayer('Name','tanh')
    
    ];

lgraphD1 = layerGraph(layersD1);
dlnetD1 = dlnetwork(lgraphD1);

% Inverse generator (IG)
load("dlnetIG1");


%% Network structure (Response2)
% Encoder (E)
layersE2 = [
    
    imageInputLayer([size(response2_source,1) 1 1],'Normalization','none','Name','in')
    
    convolution2dLayer(filterSize,8,'Stride',2,'Padding',0,'Name','conv1')
    reluLayer('Name','relu1')
    
    convolution2dLayer(filterSize,16,'Stride',2,'Padding',0,'Name','conv2')
    reluLayer('Name','relu2')
    
    convolution2dLayer(filterSize,32,'Stride',2,'Padding',0,'Name','conv3')
    reluLayer('Name','relu3')
    
    convolution2dLayer(filterSize,64,'Stride',2,'Padding',0,'Name','conv4')
    reluLayer('Name','relu4')
    
    fullyConnectedLayer(2*latentDim,'Name','fc_encoder')
    
    ];

lgraphE2 = layerGraph(layersE2);
dlnetE2 = dlnetwork(lgraphE2);

% Decoder (D)
layersD2 = [
    
    imageInputLayer([1 1 latentDim],'Normalization','none','Name','in')
    
    transposedConv2dLayer(filterSize,64,'Name','tcov1')
    reluLayer('Name','relu1')
    
    transposedConv2dLayer(filterSize,32,'Stride',2,'Cropping',0,'Name','tcov2')
    reluLayer('Name','relu2')
    
    transposedConv2dLayer(filterSize,16,'Stride',2,'Cropping',0,'Name','tcov3')
    reluLayer('Name','relu3')
    
    transposedConv2dLayer(filterSize,8,'Stride',2,'Cropping',0,'Name','tcov4')
    reluLayer('Name','relu4')
    
    transposedConv2dLayer(filterSize,1,'Stride',2,'Cropping',0,'Name','tcov5')
    tanhLayer('Name','tanh')
    
    ];

lgraphD2 = layerGraph(layersD2);
dlnetD2 = dlnetwork(lgraphD2);

% Inverse generator (IG)
load("dlnetIG2");


%% Specify training options
epoch = 1000;
miniBatchSize = 50;
augimds1.MiniBatchSize = miniBatchSize;
augimds2.MiniBatchSize = miniBatchSize;
learnRateE = 0.001;
learnRateD = 0.001;


%% Train model
iteration = 0;
start = tic;
figure(1); 
figure(2); 
figure(3)
lossAxes1 = subplot(1,1,1);
lineLoss1 = animatedline(lossAxes1,'Color',[1 0 0]);
figure(4)
lossAxes2 = subplot(1,1,1);
lineLoss2 = animatedline(lossAxes2,'Color',[1 0 0]);

avgGradientsE1 = []; avgGradientsSquaredE1 = []; avgGradientsD1 = []; avgGradientsSquaredD1 = [];
avgGradientsE2 = []; avgGradientsSquaredE2 = []; avgGradientsD2 = []; avgGradientsSquaredD2 = [];
lossELBO1 = []; lossELBO2 = [];

% Loop over epochs
for i = 1:epoch
                              
    % Shuffle datastore    
    augimds1 = shuffle(augimds1);
    augimds2 = shuffle(augimds2);
    
    % Loop over mini-batches
    while hasdata(augimds1)
        iteration = iteration + 1;

        % Read mini-batch of data.
        data1 = read(augimds1);
        data2 = read(augimds2);
        
        % Ignore last partial mini-batch of epoch
        if size(data1,1) < miniBatchSize
            continue
        end
        
        if size(data2,1) < miniBatchSize
            continue
        end
        
        % Concatenate mini-batch of data and generate latent inputs for the decoder network
        y1_training = cat(4,data1{:,1}{:});
        min_y1 = min(response1_value_source(:)); max_y1 = max(response1_value_source(:));        
        y1_normalize_training = y1_training;
        
        y2_training = cat(4,data2{:,1}{:});
        min_y2 = min(response2_value_source(:)); max_y2 = max(response2_value_source(:));
        y2_normalize_training = rescale(y2_training,-1,1,'InputMin',min_y2,'InputMax',max_y2);
        
        x1_training = randn(1,size(y1_training,4))';
        x2_training = randn(1,size(y1_training,4))';
        x3_training = randn(1,size(y1_training,4))';
        x4_training = randn(1,size(y1_training,4))';
        x5_training = randn(1,size(y1_training,4))';
                
        for j = 1:size(y1_training,4)
            x_training(1,j) = [x1_training(j,1)];
            x_training(2,j) = [x2_training(j,1)];
            x_training(3,j) = [x3_training(j,1)];
            x_training(4,j) = [x4_training(j,1)];
            x_training(5,j) = [x5_training(j,1)];
            x_4D_training(:,:,1,j) = [x_training(1,j)];
            x_4D_training(:,:,2,j) = [x_training(2,j)];
            x_4D_training(:,:,3,j) = [x_training(3,j)];
            x_4D_training(:,:,4,j) = [x_training(4,j)];
            x_4D_training(:,:,5,j) = [x_training(5,j)];
        end
                
        % Convert mini-batch of data to dlarray specify the dimension labels 'SSCB' (spatial, spatial, channel, batch)
        dly1_training = dlarray(y1_normalize_training,'SSCB');
        dly2_training = dlarray(y2_normalize_training,'SSCB');
        dlx_training = dlarray(x_4D_training,'SSCB');
                
        % Evaluate the model gradients using dlfeval and the modelGradients function
        [gradientsE1,gradientsD1,ELBO1] = ...
            dlfeval(@modelGradients,dlnetE1,dlnetD1,dly1_training,dlx_training);
        
        [gradientsE2,gradientsD2,ELBO2] = ...
            dlfeval(@modelGradients,dlnetE2,dlnetD2,dly2_training,dlx_training);
                
        % Update the encoder network parameters
        [dlnetE1.Learnables,avgGradientsE1,avgGradientsSquaredE1] = ...
            adamupdate(dlnetE1.Learnables,gradientsE1,avgGradientsE1,avgGradientsSquaredE1,iteration,learnRateE);
        
        [dlnetE2.Learnables,avgGradientsE2,avgGradientsSquaredE2] = ...
            adamupdate(dlnetE2.Learnables,gradientsE2,avgGradientsE2,avgGradientsSquaredE2,iteration,learnRateE);
                
        [dlz1_training,z1_mean,z1_logVar] = sampling(dlnetE1,dly1_training,x_training);
        [dlz2_training,z2_mean,z2_logVar] = sampling(dlnetE2,dly2_training,x_training);

        % Update the decoder network parameters
        [dlnetD1.Learnables,avgGradientsD1,avgGradientsSquaredD1] = ...
            adamupdate(dlnetD1.Learnables,gradientsD1,avgGradientsD1,avgGradientsSquaredD1,iteration,learnRateD);
        
        [dlnetD2.Learnables,avgGradientsD2,avgGradientsSquaredD2] = ...
            adamupdate(dlnetD2.Learnables,gradientsD2,avgGradientsD2,avgGradientsSquaredD2,iteration,learnRateD);               
                  
        % Every 1 iterations, display batch of generated images using the held-out decoder input
        if mod(iteration,1) == 0 || iteration == 1
                   
            % Generate data using the decoder input          
            dly1_prediction_training = predict(dlnetD1,dlz1_training);
            dly2_prediction_training = predict(dlnetD2,dlz2_training);
                                                
            % Rescale the data and display the data            
            y1_extraction_training = extractdata(dly1_prediction_training(:,1));
            response1_value_training = y1_extraction_training;            
            y2_extraction_training = extractdata(dly2_prediction_training(:,1));
            response2_value_training = rescale(y2_extraction_training,min_y2,max_y2,'InputMin',-1,'InputMax',1);
                                    
            figure(1)
            movegui([350 550]);
            plot(response1_time_source(:,1),response1_value_training,'k');
            xlim([0 102.3]); ylim([-0.2 0.6]);
            set(gca,'fontsize',15,'fontname','times new roman');
            xlabel('Time [s]','fontsize',25,'fontname','times new roman'); 
            ylabel('Vibration acceleration [m/s^2]','fontsize',25,'fontname','times new roman');            
            
            % Update the title with training progress information
            D = duration(0,0,toc(start),'Format','hh:mm:ss');
            title(...
                "Epoch: " + i + ", " + ...
                "Iteration: " + iteration + ", " + ...
                "Elapsed: " + string(D))
            
            figure(2)
            movegui([950 550]);
            plot(response2_time_source(:,1),response2_value_training,'k');
            xlim([0 102.3]); ylim([-0.2 0.2]);
            set(gca,'fontsize',15,'fontname','times new roman');
            xlabel('Time [s]','fontsize',25,'fontname','times new roman');
            ylabel('Lateral deviation [m]','fontsize',25,'fontname','times new roman');
                        
            % Update the title with training progress information
            D = duration(0,0,toc(start),'Format','hh:mm:ss');
            title(...
                "Epoch: " + i + ", " + ...
                "Iteration: " + iteration + ", " + ...
                "Elapsed: " + string(D)) 
          
            % Update the losses plot
            figure(3)
            movegui([350 30]);
            subplot(1,1,1)            
            addpoints(lineLoss1,iteration,double(gather(extractdata(ELBO1))));
            legend('ELBO');
            set(gca,'fontsize',15,'fontname','times new roman');
            xlabel('Iteration','fontsize',25,'fontname','times new roman')
            ylabel('Loss','fontsize',25,'fontname','times new roman')
            grid on
                                
            % Update the title with training progress information
            D = duration(0,0,toc(start),'Format','hh:mm:ss');
            title(...
                "Epoch: " + i + ", " + ...
                "Iteration: " + iteration + ", " + ...
                "Elapsed: " + string(D))                               
            drawnow
            
            figure(4)
            movegui([950 30]);
            subplot(1,1,1)            
            addpoints(lineLoss2,iteration,double(gather(extractdata(ELBO2))));
            legend('ELBO');
            set(gca,'fontsize',15,'fontname','times new roman');
            xlabel('Iteration','fontsize',25,'fontname','times new roman')
            ylabel('Loss','fontsize',25,'fontname','times new roman')
            grid on
                                
            % Update the title with training progress information
            D = duration(0,0,toc(start),'Format','hh:mm:ss');
            title(...
                "Epoch: " + i + ", " + ...
                "Iteration: " + iteration + ", " + ...
                "Elapsed: " + string(D))                               
            drawnow
            
        end
    end                   

    ELBOLoss1 = double(gather(extractdata(ELBO1)));
    ELBOLoss2 = double(gather(extractdata(ELBO2)));
        
    lossELBO1 = [lossELBO1; ELBOLoss1];
    lossELBO2 = [lossELBO2; ELBOLoss2];
        
    disp("Epoch: " + i + ", " + "lossELBO1: " + ELBOLoss1 + ", " + "lossELBO2: " + ELBOLoss2);
end


%% Generate new data
numGenerationData = 100;

x1_test = randn(1,1,1,numGenerationData);
x2_test = randn(1,1,1,numGenerationData);
x3_test = randn(1,1,1,numGenerationData);
x4_test = randn(1,1,1,numGenerationData);
x5_test = randn(1,1,1,numGenerationData);

for i = 1:numGenerationData
    rand_number = randi([1 miniBatchSize],1,1);    
    z1_test(:,:,1,i) = [extractdata(z1_mean(1,rand_number)) + x1_test(:,:,1,i)*(exp(.5 * (extractdata(z1_logVar(1,rand_number)))))];
    z1_test(:,:,2,i) = [extractdata(z1_mean(2,rand_number)) + x2_test(:,:,1,i)*(exp(.5 * (extractdata(z1_logVar(2,rand_number)))))];
    z1_test(:,:,3,i) = [extractdata(z1_mean(3,rand_number)) + x3_test(:,:,1,i)*(exp(.5 * (extractdata(z1_logVar(3,rand_number)))))];
    z1_test(:,:,4,i) = [extractdata(z1_mean(4,rand_number)) + x4_test(:,:,1,i)*(exp(.5 * (extractdata(z1_logVar(4,rand_number)))))];
    z1_test(:,:,5,i) = [extractdata(z1_mean(5,rand_number)) + x5_test(:,:,1,i)*(exp(.5 * (extractdata(z1_logVar(5,rand_number)))))];
    z2_test(:,:,1,i) = [extractdata(z2_mean(1,rand_number)) + x1_test(:,:,1,i)*(exp(.5 * (extractdata(z2_logVar(1,rand_number)))))];
    z2_test(:,:,2,i) = [extractdata(z2_mean(2,rand_number)) + x2_test(:,:,1,i)*(exp(.5 * (extractdata(z2_logVar(2,rand_number)))))];
    z2_test(:,:,3,i) = [extractdata(z2_mean(3,rand_number)) + x3_test(:,:,1,i)*(exp(.5 * (extractdata(z2_logVar(3,rand_number)))))];
    z2_test(:,:,4,i) = [extractdata(z2_mean(4,rand_number)) + x4_test(:,:,1,i)*(exp(.5 * (extractdata(z2_logVar(4,rand_number)))))];
    z2_test(:,:,5,i) = [extractdata(z2_mean(5,rand_number)) + x5_test(:,:,1,i)*(exp(.5 * (extractdata(z2_logVar(5,rand_number)))))];
end

dlz1_test = dlarray(z1_test,'SSCB');
dlz2_test = dlarray(z2_test,'SSCB');

% Generate data using the decoder input
dly1_prediction_test = predict(dlnetD1,dlz1_test);
dly2_prediction_test = predict(dlnetD2,dlz2_test);

% Rescale the data and display the data (Response1)
for i = 1:numGenerationData
    y1_extraction_test = extractdata(dly1_prediction_test(:,:,:,i));
    response1_value_test(:,:,1,i) = y1_extraction_test;
end

response1_value_target = []; response1_target = [];

for i = 1:numGenerationData
    response1_time_target = response1_source(:,1).*ones(size(response1_source,1),numGenerationData);
    response1_value_target = [response1_value_target response1_value_test(:,:,1,i)];
    response1_target = [response1_target response1_time_target(:,i) response1_value_target(:,i)];
end

% Rescale the data and display the data (Response2)
for i = 1:numGenerationData
    y2_extraction_test = extractdata(dly2_prediction_test(:,:,:,i));
    response2_value_test(:,:,1,i) = rescale(y2_extraction_test,min_y2,max_y2,'InputMin',-1,'InputMax',1);
end

response2_value_target = []; response2_target = [];

for i = 1:numGenerationData
    response2_time_target = response2_source(:,1).*ones(size(response2_source,1),numGenerationData);
    response2_value_target = [response2_value_target response2_value_test(:,:,1,i)];
    response2_target = [response2_target response2_time_target(:,i) response2_value_target(:,i)];
end

% Estimate the design variables
dlx_predictionIG1_test = predict(dlnetIG1,response1_value_test);
dlx_predictionIG2_test = predict(dlnetIG2,response2_value_test);

for i = 1:numGenerationData
    x_mean_test(:,:,1,i) = [mean([dlx_predictionIG1_test(i,1),dlx_predictionIG2_test(i,1)]) mean([dlx_predictionIG1_test(i,2),dlx_predictionIG2_test(i,2)]) mean([dlx_predictionIG1_test(i,3),dlx_predictionIG2_test(i,3)]) mean([dlx_predictionIG1_test(i,4),dlx_predictionIG2_test(i,4)]) mean([dlx_predictionIG1_test(i,5),dlx_predictionIG2_test(i,5)])];
end

x1_mean_test = []; x2_mean_test = []; x3_mean_test = []; x4_mean_test = []; x5_mean_test = [];

for i = 1:numGenerationData    
    x1_mean_test = [x1_mean_test x_mean_test(:,1,1,i)];
    x2_mean_test = [x2_mean_test x_mean_test(:,2,1,i)];    
    x3_mean_test = [x3_mean_test x_mean_test(:,3,1,i)];    
    x4_mean_test = [x4_mean_test x_mean_test(:,4,1,i)];    
    x5_mean_test = [x5_mean_test x_mean_test(:,5,1,i)];    
end

x1_target = x1_mean_test';
x2_target = x2_mean_test';
x3_target = x3_mean_test';
x4_target = x4_mean_test';
x5_target = x5_mean_test';
x_target = [x1_target x2_target x3_target x4_target x5_target];

save dlnetD1.mat dlnetD1
save dlnetD2.mat dlnetD2


%% Optimization (GA)
load("response1_initial_source");
load("response2_initial_source");

WIFac1 = WIFac(response1_initial_source,response1_target);
WIFac2 = WIFac(response2_initial_source,response2_target);

for i = 1:numGenerationData
    WIFac_mean(i,1) = (WIFac1(i,1)+WIFac2(i,1))/2;
end

WIFac_sort = sort(WIFac_mean,'descend');

fval_target = [];
x_opt_target = [];
response1_value_target = [];
response2_value_target = [];

for i = 1:10
i
load("dlnetD1");
load("dlnetD2");

[row] = find(WIFac_mean==WIFac_sort(i));
x0 = x_target(row,:);
x_lb = [4500 80 1800 0 450];
x_ub = [5500 120 2200 0.5 550];
w = [1 1 1 1 1];
w_lb = [0.999 0.999 0.999 0.999 0.999];
w_ub = [1 1 1 1 1];

fcn = @(w)[myfun1(w,x0,x_lb,x_ub,min_y1,max_y1), myfun2(w,x0,x_lb,x_ub,min_y2,max_y2)];

options = optimoptions('gamultiobj',"PlotFcn","gaplotpareto",'PopulationSize',10,'Generations',100,'Display','iter','InitialPopulationMatrix',w);

[w_opt,fval] = gamultiobj(fcn,5,[],[],[],[],w_lb,w_ub,[],options);

fval_target = [fval_target; fval];

for i = 1:5
    x_scale(:,:,i,1) = rescale(x0(:,i),0,1,'InputMin',x_lb(:,i),'InputMax',x_ub(:,i));
end

dlx_scale = dlarray(x_scale,'SSCB');

dly1_prediction = predict(dlnetD1,dlx_scale);
dly2_prediction = predict(dlnetD2,dlx_scale);

y1_extraction = extractdata(dly1_prediction(:,1));
response1_value = y1_extraction;
response1_value_target = [response1_value_target response1_value];

y2_extraction = extractdata(dly2_prediction(:,1));
response2_value = rescale(y2_extraction,min_y2,max_y2,'InputMin',-1,'InputMax',1);
response2_value_target = [response2_value_target response2_value];

figure(5)
p1_response1 = plot(response1_time_source(:,1),response1_initial_source(:,2),'k','LineWidth',2); hold on;
p2_response1 = plot(response1_time_source(:,1),response1_value,'b','LineWidth',2);
xlim([0 102.3]); ylim([-0.2 0.6]);
set(gca,'fontsize',15,'fontname','times new roman');
xlabel('Time [s]','fontsize',25,'fontname','times new roman');
ylabel('Vibration acceleration [m/s^2]','fontsize',25,'fontname','times new roman');
title("Optimized Data",'fontsize',20,'fontname','times new roman');
legend([p1_response1 p2_response1],{'Source','Target (DVAE)'},'Location','northeast');

figure(6)
p1_response2 = plot(response2_time_source(:,1),response2_initial_source(:,2),'k','LineWidth',2); hold on;
p2_response2 = plot(response2_time_source(:,1),response2_value,'b','LineWidth',2);
xlim([0 102.3]); ylim([-0.2 0.3]);
set(gca,'fontsize',15,'fontname','times new roman');
xlabel('Time [s]','fontsize',25,'fontname','times new roman');
ylabel('Lateral deviation [m]','fontsize',25,'fontname','times new roman');
title("Optimized Data",'fontsize',20,'fontname','times new roman');
legend([p1_response2 p2_response2],{'Source','Target (DVAE)'},'Location','northeast');

dlx_predictionIG1_opt = predict(dlnetIG1,response1_value);
dlx_predictionIG2_opt = predict(dlnetIG2,response2_value);

x1_predictionIG1_opt = dlx_predictionIG1_opt(:,1,1,1);
x2_predictionIG1_opt = dlx_predictionIG1_opt(:,2,1,1);
x3_predictionIG1_opt = dlx_predictionIG1_opt(:,3,1,1);
x4_predictionIG1_opt = dlx_predictionIG1_opt(:,4,1,1);
x5_predictionIG1_opt = dlx_predictionIG1_opt(:,5,1,1);
x1_predictionIG2_opt = dlx_predictionIG2_opt(:,1,1,1);
x2_predictionIG2_opt = dlx_predictionIG2_opt(:,2,1,1);
x3_predictionIG2_opt = dlx_predictionIG2_opt(:,3,1,1);
x4_predictionIG2_opt = dlx_predictionIG2_opt(:,4,1,1);
x5_predictionIG2_opt = dlx_predictionIG2_opt(:,5,1,1);

x1_IG1_opt_target = rescale(x1_predictionIG1_opt,x_lb(1),x_ub(1),'InputMin',x_lb(1),'InputMax',x_ub(1));
x2_IG1_opt_target = rescale(x2_predictionIG1_opt,x_lb(2),x_ub(2),'InputMin',x_lb(2),'InputMax',x_ub(2));
x3_IG1_opt_target = rescale(x3_predictionIG1_opt,x_lb(3),x_ub(3),'InputMin',x_lb(3),'InputMax',x_ub(3));
x4_IG1_opt_target = rescale(x4_predictionIG1_opt,x_lb(4),x_ub(4),'InputMin',x_lb(4),'InputMax',x_ub(4));
x5_IG1_opt_target = rescale(x5_predictionIG1_opt,x_lb(5),x_ub(5),'InputMin',x_lb(5),'InputMax',x_ub(5));
x1_IG2_opt_target = rescale(x1_predictionIG2_opt,x_lb(1),x_ub(1),'InputMin',x_lb(1),'InputMax',x_ub(1));
x2_IG2_opt_target = rescale(x2_predictionIG2_opt,x_lb(2),x_ub(2),'InputMin',x_lb(2),'InputMax',x_ub(2));
x3_IG2_opt_target = rescale(x3_predictionIG2_opt,x_lb(3),x_ub(3),'InputMin',x_lb(3),'InputMax',x_ub(3));
x4_IG2_opt_target = rescale(x4_predictionIG2_opt,x_lb(4),x_ub(4),'InputMin',x_lb(4),'InputMax',x_ub(4));
x5_IG2_opt_target = rescale(x5_predictionIG2_opt,x_lb(5),x_ub(5),'InputMin',x_lb(5),'InputMax',x_ub(5));

x_opt_target = [x_opt_target; mean([x1_IG1_opt_target,x1_IG2_opt_target]) mean([x2_IG1_opt_target,x2_IG2_opt_target]) mean([x3_IG1_opt_target,x3_IG2_opt_target]) mean([x4_IG1_opt_target,x4_IG2_opt_target]) mean([x5_IG1_opt_target,x5_IG2_opt_target])];
end


%% Optimal design variables of target
figure(7)
A1 = mean(x_opt_target(:,1));
B1 = std(x_opt_target(:,1));
Lower1 = x_lb(1);
Upper1 = x_ub(1);
pd1 = makedist('Normal',A1,B1);
t1 = truncate(pd1,Lower1,Upper1);
z1 = random(t1,100,1);
result_z1 = sort(z1);
C1 = min(result_z1);
D1 = max(result_z1);
h1 = histogram(result_z1,5,'Normalization','pdf'); hold on;
h1.FaceColor = 'b';
stem(Lower1,max(h1.Values),':.k','Linewidth',5); hold on;
stem(Upper1,max(h1.Values),':.k','Linewidth',5);
xlim([4450 5550]);
ylim([0 max(h1.Values)])
set(gca,'fontsize',15,'fontname','times new roman');
xlabel('z_{1}','fontsize',25,'fontname','times new roman'); 
ylabel('PDF','fontsize',25,'fontname','times new roman');

figure(8)
A2 = mean(x_opt_target(:,2));
B2 = std(x_opt_target(:,2));
Lower2 = x_lb(2);
Upper2 = x_ub(2);
pd2 = makedist('Normal',A2,B2);
t2 = truncate(pd2,Lower2,Upper2);
z2 = random(t2,100,1);
result_z2 = sort(z2);
C2 = min(result_z2);
D2 = max(result_z2);
h2 = histogram(result_z2,5,'Normalization','pdf'); hold on;
h2.FaceColor = 'b';
stem(Lower2,max(h2.Values),':.k','Linewidth',5); hold on;
stem(Upper2,max(h2.Values),':.k','Linewidth',5);
xlim([70 130]);
ylim([0 max(h2.Values)])
set(gca,'fontsize',15,'fontname','times new roman');
xlabel('z_{2}','fontsize',25,'fontname','times new roman'); 
ylabel('PDF','fontsize',25,'fontname','times new roman'); 

figure(9)
A3 = mean(x_opt_target(:,3));
B3 = std(x_opt_target(:,3));
Lower3 = x_lb(3);
Upper3 = x_ub(3);
pd3 = makedist('Normal',A3,B3);
t3 = truncate(pd3,Lower3,Upper3);
z3 = random(t3,100,1);
result_z3 = sort(z3);
C3 = min(result_z3);
D3 = max(result_z3);
h3 = histogram(result_z3,5,'Normalization','pdf'); hold on;
h3.FaceColor = 'b';
stem(Lower3,max(h3.Values),':.k','Linewidth',5); hold on;
stem(Upper3,max(h3.Values),':.k','Linewidth',5);
xlim([1750 2250]);
ylim([0 max(h3.Values)])
set(gca,'fontsize',15,'fontname','times new roman');
xlabel('z_{3}','fontsize',25,'fontname','times new roman'); 
ylabel('PDF','fontsize',25,'fontname','times new roman'); 

figure(10)
A4 = mean(x_opt_target(:,4));
B4 = std(x_opt_target(:,4));
Lower4 = x_lb(4);
Upper4 = x_ub(4);
pd4 = makedist('Normal',A4,B4);
t4 = truncate(pd4,Lower4,Upper4);
z4 = random(t4,100,1);
result_z4 = sort(z4);
C4 = min(result_z4);
D4 = max(result_z4);
h4 = histogram(result_z4,5,'Normalization','pdf'); hold on;
h4.FaceColor = 'b';
stem(Lower4,max(h4.Values),':.k','Linewidth',5); hold on;
stem(Upper4,max(h4.Values),':.k','Linewidth',5);
xlim([-0.1 0.6]);
ylim([0 max(h4.Values)])
set(gca,'fontsize',15,'fontname','times new roman');
xlabel('z_{4}','fontsize',25,'fontname','times new roman'); 
ylabel('PDF','fontsize',25,'fontname','times new roman'); 

figure(11)
A5 = mean(x_opt_target(:,5));
B5 = std(x_opt_target(:,5));
Lower5 = x_lb(5);
Upper5 = x_ub(5);
pd5 = makedist('Normal',A5,B5);
t5 = truncate(pd5,Lower5,Upper5);
z5 = random(t5,100,1);
result_z5 = sort(z5);
C5 = min(result_z5);
D5 = max(result_z5);
h5 = histogram(result_z5,5,'Normalization','pdf'); hold on;
h5.FaceColor = 'b';
stem(Lower5,max(h5.Values),':.k','Linewidth',5); hold on;
stem(Upper5,max(h5.Values),':.k','Linewidth',5);
xlim([400 600]);
ylim([0 max(h5.Values)])
set(gca,'fontsize',15,'fontname','times new roman');
xlabel('z_{5}','fontsize',25,'fontname','times new roman'); 
ylabel('PDF','fontsize',25,'fontname','times new roman'); 


%% Objective function 1
function f = myfun1(w,x0,x_lb,x_ub,min_Y_s,max_Y_s)

global dlnetD1

dlnetD1.Learnables.Value{1,1}(:,:,1,:) = w(1)*dlnetD1.Learnables.Value{1,1}(:,:,1,:);
dlnetD1.Learnables.Value{3,1}(:,:,1,:) = w(2)*dlnetD1.Learnables.Value{3,1}(:,:,1,:);
dlnetD1.Learnables.Value{5,1}(:,:,1,:) = w(3)*dlnetD1.Learnables.Value{5,1}(:,:,1,:);
dlnetD1.Learnables.Value{7,1}(:,:,1,:) = w(4)*dlnetD1.Learnables.Value{7,1}(:,:,1,:);
dlnetD1.Learnables.Value{9,1}(:,:,1,:) = w(5)*dlnetD1.Learnables.Value{9,1}(:,:,1,:);

for i = 1:5
    scale_X(:,:,i,1) = rescale(x0(:,i),0,1,'InputMin',x_lb(:,i),'InputMax',x_ub(:,i));
end

dlX_Training = dlarray(scale_X,'SSCB');

dlYPred = predict(dlnetD1,dlX_Training);

Y_Normalize = extractdata(dlYPred(:,1));

Y_DeNormalize = Y_Normalize;

Fs= 10;
L = size(Y_DeNormalize,1);
Y = fft(Y_DeNormalize);
P2 = abs(Y/L);
P1 = P2(1:L/2+1);
P1(2:end-1) = 2*P1(2:end-1);
f = max(P1);
end


%% Objective Function 2
function f = myfun2(w,x0,x_lb,x_ub,min_y,max_y)

global dlnetD2

dlnetD2.Learnables.Value{1,1}(:,:,1,:) = w(1)*dlnetD2.Learnables.Value{1,1}(:,:,1,:);
dlnetD2.Learnables.Value{3,1}(:,:,1,:) = w(2)*dlnetD2.Learnables.Value{3,1}(:,:,1,:);
dlnetD2.Learnables.Value{5,1}(:,:,1,:) = w(3)*dlnetD2.Learnables.Value{5,1}(:,:,1,:);
dlnetD2.Learnables.Value{7,1}(:,:,1,:) = w(4)*dlnetD2.Learnables.Value{7,1}(:,:,1,:);
dlnetD2.Learnables.Value{9,1}(:,:,1,:) = w(5)*dlnetD2.Learnables.Value{9,1}(:,:,1,:);

for i = 1:5
    scale_X(:,:,i,1) = rescale(x0(:,i),0,1,'InputMin',x_lb(:,i),'InputMax',x_ub(:,i));
end

dlX_Training = dlarray(scale_X,'SSCB');

dlYPred = predict(dlnetD2,dlX_Training);

Y_Normalize = extractdata(dlYPred(:,:,:,1));

Y_DeNormalize = rescale(Y_Normalize,min_y,max_y,'InputMin',-1,'InputMax',1);

Fs= 10;
L = size(Y_DeNormalize,1);
Y = fft(Y_DeNormalize);
P2 = abs(Y/L);
P1 = P2(1:L/2+1);
P1(2:end-1) = 2*P1(2:end-1);
f = max(P1);
end


%% Model gradients function
function [infGrad,genGrad,loss] = modelGradients(dlnetEncoder,dlnetDecoder,Y,X)
[X,XMean,XLogvar] = sampling(dlnetEncoder,Y,X);
YPred = forward(dlnetDecoder,X);
loss = ELBOloss(Y,YPred,XMean,XLogvar);
[genGrad,infGrad] = dlgradient(loss,dlnetDecoder.Learnables,dlnetEncoder.Learnables);
end


%% Sampling function
function [zSampled,zMean,zLogvar] = sampling(dlnetEncoder,Y,X)
compressed = forward(dlnetEncoder,Y);
d = size(compressed,1)/2;
zMean = compressed(1:d,:);
zLogvar = compressed(1+d:end,:);
sz = size(zMean);
epsilon = X;
sigma = exp(.5 * zLogvar);
z = epsilon .* sigma + zMean;
z = reshape(z,[1,1,sz]);
zSampled = dlarray(z,'SSCB');
end


%% Loss function
function ELBO = ELBOloss(Y,YPred,XMean,XLogvar)

squares = 0.5*(YPred-Y).^2;
reconstructionLoss  = sum(squares,[1,2,3]);

KL = -.5 * sum(1 + XLogvar - XMean.^2 - exp(XLogvar),1);

ELBO = mean(reconstructionLoss + KL);
end