close all
clear all
rng(0);

%% Parameters
sigma = 0.01;
downsamplePercentage = 0.05;
overlap = 0.5;
dThresh = 0.1;
dPerc = 0.1;

fractionOutliers = 0.1;

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

%% Add outliers

Noutliers = fractionOutliers * length(P1);
a = 4 * max(max(abs(P1 - mean(P1))));
O1 = a * (rand(Noutliers, 3) - 0.5) + mean(P1);
O2 = a * (rand(Noutliers, 3) - 0.5) + mean(P2);

P1 = [P1; O1];
P2 = [P2; O2];

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

D1sort(D1sort > dThresh) = 0;
D2sort(D2sort > dThresh) = 0;

% D1sort = D1sort(1:nDist, :);
% D2sort = D2sort(1:nDist, :);

%% Find matches

matches = zeros(nPoints, 1);
errs = zeros(nPoints, 1);
for n = 1:nPoints
  err = inf;
  for m = 1:nPoints
    currErr = sum((D1sort(:, n) - D2sort(:, m)).^2);
    if currErr < err
      matches(n) = m;
      err = currErr;
      errs(n) = err;
    end
  end
end
      
%% Prune the matches
errs(errs < 1e-5) = [];
goodMatches = errs < quantile(errs, dPerc);
matches = matches(goodMatches);
    
%% Calculate number of errors

gt = 1:nPoints;
nErrs = sum(gt(goodMatches) ~= permi(matches));
frac = nErrs / length(matches);

disp(['Number of incorrect matches: ', num2str(nErrs)]);
disp(['Fraction of incorrect matches: ', num2str(frac)]);

%% Register the point clouds using closed form solution to orthogonal proscrustes problem
X1matched = X1m(goodMatches, :);
X2matched = X2m(matches, :);

M = X1matched' * X2matched;
[U, ~, V] = svd(M);
S = eye(3, 3);
S(3, 3) = sign(det(U * V'));
Rhat = U * S * V';
mu1 = mean(X1(goodMatches, :));
mu2 = mean(X2(matches, :));
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

plot3(X1(goodMatches, 1), X1(goodMatches, 2), X1(goodMatches, 3), 'r.')
view(2)

%% Compare with ICP

tform = pcregrigid(pointCloud(X2), pointCloud(X1), 'Extrapolate', true);


disp('Estimated transform from ICP : ');
disp(invert(tform).T);

figure
pcshow(pointCloud(X1));
hold on
pcshow(pctransform(pointCloud(X2), tform))
title('Registered point clouds using ICP');
view(2)
