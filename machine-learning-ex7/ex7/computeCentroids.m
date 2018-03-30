function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returns the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m, n] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, n);


% ====================== YOUR CODE HERE ======================
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%
% Note: You can use a for-loop over the centroids to compute this.
%

tmpTotal = zeros(K,n); % this would be used to calculate the mean of each cluster as new centroids
counter = zeros(K,1);  % calculate the number of each cluster
for i=1:m
    tmpTotal(idx(i), :) = tmpTotal(idx(i), :) + X(i, :);
    counter(idx(i)) = counter(idx(i)) + 1;
end

for j=1:K
    centroids(j, :) = tmpTotal(j, :) / counter(j);
end

% =============================================================


end

