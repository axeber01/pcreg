close all
clear all
rng(0);

%% Parameters
sigma = 0.01;
downsamplePercentage = 0.05;
overlap = 0.5;

%% Load the teapot example

ptCloud = pcread('teapot.ply');

%% Downsample
ptCloud = pcdownsample(ptCloud, 'random', downsamplePercentage);

%% Create overlap
P1 = ptCloud.Location;
P2 = ptCloud.Location;

P1(P1(:, 2) > 1, :) = [];
P2(P2(:, 2) < -1, :) = [];

P1(length(P2)+1:end, :) = [];

%%

figure
pcshow(pointCloud(P1)); 
title('Teapot');

%% Create transformed ptcloud (only rotation)

angles = [pi/6, pi/3, pi];
translation = [5, 5, 10];
R = eul2rotm(angles);

A = [R; translation];
A = [A, [0; 0; 0; 1]];

tform1 = affine3d(A);

X1 = P1;
X2 = pctransform(pointCloud(P2),tform1).Location;
nPoints = size(X1, 1);

% Apply random permutation on X2
permi = randperm(nPoints);
X2 = X2(permi, :);

% Add Gaussian noise
X2 = X2 + sigma * randn(nPoints, 3);

hold on
pcshow(pointCloud(X2));

%% Find centroids
mu1 = mean(X1);
mu2 = mean(X2);

X1m = X1 - mu1;
X2m = X2 - mu2;

%% Creat distance matrices
o = ones(nPoints, 1);

z1 = sum(X1m'.^2)';
D1 = o * z1' + z1 * o' - 2 * X1m * X1m';

z2 = sum(X2m'.^2)';
D2 = o * z2' + z2 * o' - 2 * X2m * X2m';


%% Sort the columns of distance matrices

[D1sort, i1] = sort(D1);
[D2sort, i2] = sort(D2);

%% Find matches

matches = zeros(nPoints, 1);
for n = 1:nPoints
  err = inf;
  for m = 1:nPoints
    currErr = sum((D1sort(:, n) - D2sort(:, m)).^2);
    if currErr < err
      matches(n) = m;
      err = currErr;
    end
  end
end
      
    
%% Calculate number of errors

gt = 1:nPoints;
nErrs = sum(gt ~= permi(matches));
frac = nErrs / nPoints;

disp(['Number of incorrect matches: ', num2str(nErrs)]);
disp(['Fraction of incorrect matches: ', num2str(frac)]);

%% Register the point clouds using closed form solution to orthogonal proscrustes problem
X2matched = X2m(matches, :);

M = X1m' * X2matched;
[U, ~, V] = svd(M);
S = eye(3, 3);
S(3, 3) = sign(det(U * V'));
Rhat = U * S * V';
that = mu2 - (Rhat' * mu1')';

disp('Ground truth transformation: ');
disp(A);
disp('Estimated transformation from distance matching : ');
disp([Rhat; that]);

%% Show merged pointcloud
X1hat = (Rhat * (X2-that)')';
p = pointCloud(X1);
phat = pointCloud(X1hat);

figure
pcshow(p);
hold on
pcshow(phat)
title('Registered point clouds using distance matching');

%% Compare with ICP

tform = pcregrigid(pointCloud(X2), pointCloud(X1), 'Extrapolate', true);


disp('Estimated transform from ICP : ');
disp(invert(tform).T);

figure
pcshow(pointCloud(X1));
hold on
pcshow(pctransform(pointCloud(X2), tform))
title('Registered point clouds using ICP');