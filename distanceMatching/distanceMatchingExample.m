close all
clear all
rng(0);

%% Parameters
sigma = 0.01;
downsamplePercentage = 0.05;
overlap = 0.5;
dThresh = 1;
dPerc = 0.1;
doRansac = true;
nRansac = 1000;
rThresh = 0.2;

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

% P1(length(P2)+1:end, :) = [];

%% Add outliers

Noutliers = round(fractionOutliers * length(P1));
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
nPoints = size(X2, 1);

% Apply random permutation on X2
permi = randperm(nPoints);
X2 = X2(permi, :);

% Add Gaussian noise
X2 = X2 + sigma * randn(nPoints, 3);

hold on
pcshow(pointCloud(X2));


%% Peform matching and registration
[Rhat, that, matches, goodMatches] = performDistanceMatching(X1, X2, dThresh, dPerc, doRansac, nRansac, rThresh);

disp('Ground truth transformation: ');
disp(A);
disp('Estimated transformation from distance matching : ');
disp([Rhat; that]);

%% Calculate number of errors

% gt = 1:nPoints;
% nErrs = sum(gt(goodMatches) ~= permi(matches));
% frac = nErrs / length(matches);
% 
% disp(['Number of incorrect matches: ', num2str(nErrs)]);
% disp(['Fraction of incorrect matches: ', num2str(frac)]);

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
