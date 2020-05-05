function [dlY,state,dlT] = pointnetRegressor(dlXList,parameters,state,isTraining)

% Invoke the PointNet encoder.
[dlY,state,dlT] = siamesePointnetEncoder(dlXList,parameters,state,isTraining);

% Invoke the classifier.
p = parameters.ClassificationMLP.Perceptron;
s = state.ClassificationMLP.Perceptron;
for k = 1:numel(p) 
     
    [dlY, s(k)] = perceptron(dlY,p(k),s(k),isTraining);
      
    % If training, apply inverted dropout with a probability of 0.3.
    if isTraining
        probability = 0.3; 
        dropoutScaleFactor = 1 - probability;
        dropoutMask = ( rand(size(dlY), "like", dlY) > probability ) / dropoutScaleFactor;
        dlY = dlY.*dropoutMask;
    end
    
end
state.ClassificationMLP.Perceptron = s;

% Apply final fully connected and linear operations.
weights = parameters.ClassificationMLP.FC.Weights;
bias = parameters.ClassificationMLP.FC.Bias;
dlY = fullyconnect(dlY,weights,bias);
end