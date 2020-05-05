function [params,state] = initializeRegressionMLP(inputChannelSize,hiddenChannelSize,numOutputs)
[params,state] = initializeSharedMLP(inputChannelSize,hiddenChannelSize);

weights = initializeWeightsGaussian([numOutputs hiddenChannelSize(end)]);
bias = zeros(numOutputs,1,"single");
params.FC.Weights = dlarray(weights);
params.FC.Bias = dlarray(bias);
end