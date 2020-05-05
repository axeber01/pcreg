function [dlYout,state,T] = siamesePointnetEncoder(dlXList,parameters,state,isTraining)

outSize = size(parameters.SharedMLP2.Perceptron.Conv.Weights, 4);
%dlYout = zeros(1, size(dlXList, 2), outSize, size(dlXList, 4), 'like', dlXList);

chanSize = size(dlXList, 2);
dlYout = zeros(1, 1, outSize * chanSize, size(dlXList, 4), 'like', dlXList);

dlYout = dlarray(dlYout,'SSCB');

for n = 1:chanSize
  dlX = dlXList(:, n, :, :);
  
  % Input transform.
  %[dlY,state.InputTransform] = dataTransform(dlX,parameters.InputTransform,state.InputTransform,isTraining);
  
  % Shared MLP.
  [dlY,state.SharedMLP1.Perceptron] = sharedMLP(dlX,parameters.SharedMLP1.Perceptron,state.SharedMLP1.Perceptron,isTraining);
  
  % Feature transform.
  %[dlY,state.FeatureTransform,T] = dataTransform(dlY,parameters.FeatureTransform,state.FeatureTransform,isTraining);
  T = dlarray(zeros(64, 64, 1, 128));
  
  % Shared MLP.
  [dlY,state.SharedMLP2.Perceptron] = sharedMLP(dlY,parameters.SharedMLP2.Perceptron,state.SharedMLP2.Perceptron,isTraining);
  
  % Max operation.
  dlYout(:, :, (n - 1) * outSize + 1 : n * outSize, :) = mean(dlY, 1); %max(dlY,[],1)
  
end
%dlYout = cat(2, dlYout{:});

%dlYout = (dlYout(:, 2:end, :, :) - dlYout(:, 1:end-1, :, :));

end