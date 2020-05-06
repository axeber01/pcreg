%% Train siamese network to register KITTI lidar frames

clear
close all
addpath('utils/');
addpath('layers/');

%% Define hyperparameters
Minibatchsize = 128;
nClouds = 2;
nPoints = 10e3;
Rstd = deg2rad(10);
Tstd = 3;

%%  Create datastore object

datapath = '/usr/vision/axelb/kitti/2011_09_26/';
foldsTrain = 1:12;
foldsVal = 13:15;
dsTrain = pcDataStore(datapath, foldsTrain, Minibatchsize, nClouds, nPoints, Rstd, Tstd);
dsVal = pcDataStore(datapath, foldsVal, Minibatchsize, nClouds, nPoints, Rstd, Tstd);

%% Normalize point clouds

% dsTrain = transform(dsTrain,@normalizePointCloud);
% dsVal = transform(dsVal,@normalizePointCloud);
% 
% %% Downsample point clouds
% 
% dsTrain = transform(dsTrain,@(data)samplePointCloud(data,nPoints));
% dsVal = transform(dsVal,@(data)samplePointCloud(data,nPoints));


%% Define PointNet Encoder Model Parameters

% Not used
inputChannelSize = 3;
hiddenChannelSize1 = [64,128, 128, 128];
hiddenChannelSize2 = 256; %256
[parameters.InputTransform, state.InputTransform] = initializeTransform(inputChannelSize,hiddenChannelSize1,hiddenChannelSize2);

inputChannelSize = 3;
hiddenChannelSize = [64, 64];
[parameters.SharedMLP1,state.SharedMLP1] = initializeSharedMLP(inputChannelSize,hiddenChannelSize);

% Not used
inputChannelSize = 64;
hiddenChannelSize1 = [64,128, 128, 128];
hiddenChannelSize2 = 256;
[parameters.FeatureTransform, state.FeatureTransform] = initializeTransform(inputChannelSize,hiddenChannelSize,hiddenChannelSize2);

inputChannelSize = 64;
hiddenChannelSize = [64];
[parameters.SharedMLP2,state.SharedMLP2] = initializeSharedMLP(inputChannelSize,hiddenChannelSize);


%% Define PointNet Regression Model Parameters

inputChannelSize = 64 * nClouds;
hiddenChannelSize = [512, 256];
numOutputs = 6;
[parameters.ClassificationMLP, state.ClassificationMLP] = initializeRegressionMLP(inputChannelSize,hiddenChannelSize, numOutputs);

%% Training options

numEpochs = 20;
learnRate = 0.001;
l2Regularization = 0.0001;
learnRateDropPeriod = 15;
learnRateDropFactor = 0.5;

% Adam options
gradientDecayFactor = 0.9;
squaredGradientDecayFactor = 0.999;

%% Training

avgGradients = [];
avgSquaredGradients = [];

doTraining = false;


% Use the configureTrainingProgressPlot function, listed at the end of the
% example, to initialize the training progress plot to display the training
% loss, training accuracy, and validation accuracy.
[lossPlotter,trainTransPlotter,valTransPlotter, ...
  trainRotPlotter, valRotPlotter] = initializeTrainingProgressPlot;

iteration = 0;
start = tic;
for epoch = 1:numEpochs
  
  % Reset training and validation datastores.
  reset(dsTrain);
  reset(dsVal);
  
  % Iterate through data set.
  while hasdata(dsTrain)
    iteration = iteration + 1;
    
    % Read data.
    data = read(dsTrain);
    
    % Create batch.
    [XTrain,YTrain] = batchData(data);
    
    % Evaluate the model gradients and loss using dlfeval and the
    % modelGradients function.
    [gradients, loss, state, YPredTrain] = dlfeval(@modelGradients,XTrain,YTrain,parameters,state);
    
    % L2 regularization.
    gradients = dlupdate(@(g,p) g + l2Regularization*p,gradients,parameters);
    
    % Update the network parameters using the Adam optimizer.
    [parameters, avgGradients, avgSquaredGradients] = adamupdate(parameters, gradients, ...
      avgGradients, avgSquaredGradients, iteration,...
      learnRate,gradientDecayFactor, squaredGradientDecayFactor);
    
    % Update the training progress.
    D = duration(0,0,toc(start),"Format","hh:mm:ss");
    title(lossPlotter.Parent,"Epoch: " + epoch + ", Elapsed: " + string(D))
    addpoints(lossPlotter,iteration,double(gather(extractdata(loss))))
    
    label = squeeze(extractdata(gather(YTrain)));
    t = label(4:6, :) * Tstd;
    rot = label(1:3, :) * Rstd;
    
    elabel = squeeze(extractdata(gather(YPredTrain)));
    te = elabel(4:6, :) * Tstd;
    rote = elabel(1:3, :) * Rstd;
    
    maeRotTrain = rad2deg(double(mean(abs(rot(:) - rote(:)))));
    maeTransTrain = double(mean(abs(t(:) - te(:))));
    
    addpoints(trainTransPlotter,iteration,maeTransTrain);
    addpoints(trainRotPlotter,iteration,maeRotTrain);
    drawnow
  end
  
  % Evaluate the model on validation data.
  maeRot = [];
  maeTrans = [];
  n = 1;
  while hasdata(dsVal)
    
    % Get the next batch of data.
    data = read(dsVal);
    
    % Create batch.
    [XVal,YVal, sigma] = batchData(data);
    
    % Compute label predictions.
    isTraining = false;
    YPred = pointnetRegressor(XVal,parameters,state,isTraining);
    
    label = squeeze(extractdata(gather(YVal)));
    t = label(4:6, :) * Tstd;
    rot = label(1:3, :) * Rstd;
    
    elabel = squeeze(extractdata(gather(YPred)));
    te = elabel(4:6, :) * Tstd;
    rote = elabel(1:3, :) * Rstd;
    
    maeRot(n) = rad2deg(double(mean(abs(rot(:) - rote(:)))));
    maeTrans(n) = double(mean(abs(t(:) - te(:))));
    n = n + 1;
  end
  maeRot = mean(maeRot);
  maeTrans = mean(maeTrans);
  
  % Update training progress plot with average classification accuracy.
  addpoints(valTransPlotter,iteration,maeTrans);
  addpoints(valRotPlotter,iteration,maeRot);
  
  % Upate the learning rate.
  if mod(epoch,learnRateDropPeriod) == 0
    learnRate = learnRate * learnRateDropFactor;
  end
  
  % Reset training and validation datastores.
  reset(dsTrain);
  reset(dsVal);
end

%% Visualize registration
index = 2;

figure(100)
x = squeeze(extractdata(gather(XVal(:, 1, :, index))));
y = squeeze(extractdata(gather(XVal(:, 2, :, index))));

x = pointCloud(x * sigma(1, 1, 1, index));
%x.Color = uint8([0, 0, 1]);
y = pointCloud(y * sigma(1, 1, 1, index));
%y.Color = 'r';

pcshow([x.Location(:, 1), x.Location(:, 2), x.Location(:, 3)], 'b')
hold on
pcshow([y.Location(:, 1), y.Location(:, 2), y.Location(:, 3)], 'r')
title('Before Registration')
hold off
axis equal
view(2)

figure(101)
pcshow([x.Location(:, 1), x.Location(:, 2), x.Location(:, 3)], 'b')
title('True Registration')
hold on
label = squeeze(extractdata(gather(YVal(:, :, :, index))));
t = label(4:6) * Tstd;
R = eul2rotm(label(1:3)' * Rstd);
tform = rigid3d(R', -t');
yr = pctransform(y, tform);
pcshow([yr.Location(:, 1), yr.Location(:, 2), yr.Location(:, 3)], 'r')
hold off
axis equal
view(2)

figure(102)
pcshow([x.Location(:, 1), x.Location(:, 2), x.Location(:, 3)], 'b')
title('Estimated Registration')
hold on
elabel = squeeze(extractdata(gather(YPred(:, index))));
te = elabel(4:6) * Tstd;
Re = eul2rotm(elabel(1:3)' * Rstd);
tforme = rigid3d(Re', -te');
ye = pctransform(y, tforme);
pcshow([ye.Location(:, 1), ye.Location(:, 2), ye.Location(:, 3)], 'r')
hold off
axis equal
view(2)

