function [gradients, loss, state, YPred] = modelGradients(X,Y,parameters,state)

% Execute the model function.
isTraining = true;
[YPred,state,dlT] = pointnetRegressor(X,parameters,state,isTraining);

% Add regularization term to ensure feature transform matrix is
% approximately orthogonal.
%K = size(dlT,1);
%B = size(dlT, 4);
%I = repelem(eye(K),1,1,1,B);
%dlI = dlarray(I,"SSCB");
treg = 0; %mse(dlI,dlmtimes(dlT,permute(dlT,[2 1 3 4])));
factor = 0.001;

batchSize = size(YPred, 2);
% Compute the loss.
loss = sum(sum((YPred - Y).^2)) / batchSize + factor*treg;

% Compute the parameter gradients with respect to the loss. 
gradients = dlgradient(loss, parameters);

end