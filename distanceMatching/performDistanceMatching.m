function [Rhat, that, matches, goodMatches] = performDistanceMatching(X1, X2, dThresh, dPerc, doRansac, nRansac, rThresh)

if nargin < 5
  doRansac = false;
end

if nargin < 4
  dPerc = 0.1;
end

if nargin < 3
  dThresh = 0.1;
end

%% Find centroids
mu1 = mean(X1);
mu2 = mean(X2);

X1m = X1 - mu1;
X2m = X2 - mu2;

%% Creat distance matrices
nPoints1 = size(X1, 1);
nPoints2 = size(X2, 1);

o1 = ones(nPoints1, 1);
o2 = ones(nPoints2, 1);

z1 = sum(X1m'.^2)';
D1 = o1 * z1' + z1 * o1' - 2 * X1m * X1m';

z2 = sum(X2m'.^2)';
D2 = o2 * z2' + z2 * o2' - 2 * X2m * X2m';


%% Sort the columns of distance matrices

[D1sort, ~] = sort(D1);
[D2sort, ~] = sort(D2);

D1sort(D1sort > dThresh) = 0;
D2sort(D2sort > dThresh) = 0;

%% Remove the largest distances
if nPoints1 > nPoints2
  D1sort = D1sort(1:nPoints2, :);
else
  D2sort = D2sort(1:nPoints1, :);
end


%% Find matches

matches = zeros(nPoints1, 1);
errs = zeros(nPoints1, 1);
for n = 1:nPoints1
  err = inf;
  for m = 1:nPoints2
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

%% Register the point clouds using closed form solution to orthogonal proscrustes problem
X1matched = X1m(goodMatches, :);
X2matched = X2m(matches, :);

if doRansac
  err = Inf;
  for n = 1:nRansac
    s = randsample(length(X1matched), 3);
    M = X1matched(s, :)' * X2matched(s, :);
    [U, ~, V] = svd(M);
    S = eye(3, 3);
    S(3, 3) = sign(det(U * V'));
    Rhat = U * S * V';
    temp = X1(goodMatches, :);
    mu1 = mean(temp(s, :));
    mu2 = mean(X2(matches(s), :));
    that = mu2 - (Rhat' * mu1')';
    
    % Perform registration
    X2r = (Rhat * (X2(matches, :)-that)')';
    X1r = X1(goodMatches, :);
    currErr = sum(sum(abs(X1r - X2r), 2) > rThresh);
    if currErr < err
      err = currErr;
      bestR = Rhat;
      bestt = that;
      bestMatches = matches(s);
      bestGoodMatches = goodMatches(s);
    end
    
  end
  
  Rhat = bestR;
  that = bestt;
%   matches = bestMatches;
%   goodMatches = bestGoodMatches;
else
  M = X1matched' * X2matched;
  [U, ~, V] = svd(M);
  S = eye(3, 3);
  S(3, 3) = sign(det(U * V'));
  Rhat = U * S * V';
  mu1 = mean(X1(goodMatches, :));
  mu2 = mean(X2(matches, :));
  that = mu2 - (Rhat' * mu1')';
end

end

