function [dlY,state] = sharedMLP(dlX,parameters,state,isTraining)
dlY = dlX;
for k = 1:numel(parameters) 
    [dlY, state(k)] = perceptron(dlY,parameters(k),state(k),isTraining);
end
end