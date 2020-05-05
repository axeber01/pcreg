function [dlX,dlY, sigma] = batchData(data)
X = cat(4,data{:,1});
Y = cat(4,data{:,2});
sigma = cat(4,data{:,3});

% Cast data to single for processing.
X = single(X);
Y = single(Y);

% Move data to the GPU if possible.
if canUseGPU
    X = gpuArray(X);
    Y = gpuArray(Y);
end

% Return X and Y as dlarray objects.
dlX = dlarray(X,'SSCB');
dlY = dlarray(Y,'CSSB');
end