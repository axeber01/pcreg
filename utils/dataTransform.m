function [dlY,state,T] = dataTransform(dlX,parameters,state,isTraining)

% Shared MLP.
[dlY,state.Block1.Perceptron] = sharedMLP(dlX,parameters.Block1.Perceptron,state.Block1.Perceptron,isTraining);

% Max operation.
dlY = max(dlY,[],1);

% Shared MLP.
[dlY,state.Block2.Perceptron] = sharedMLP(dlY,parameters.Block2.Perceptron,state.Block2.Perceptron,isTraining);

% Transform net (T-Net). Apply last fully connected operation as W*X to
% predict tranformation matrix T.
dlY = extractdata(dlY);
dlY = squeeze(dlY); % N-by-B
T = parameters.Transform * dlY; % K^2-by-B

% Reshape T into a square matrix.
K = sqrt(size(T,1));
T = reshape(T,K,K,1,[]); % [K K 1 B]
T = T + eye(K);

% Apply to input dlX using batch matrix multiply. 
X = extractdata(dlX); % [M 1 K B]
[C,B] = size(X,[3 4]);
X = reshape(X,[],C,1,B); % [M K 1 B]
Y = dlmtimes(X,T);
dlY = dlarray(Y,"SCSB");
end