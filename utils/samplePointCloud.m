function data = samplePointCloud(data,numPoints) 
% Select the desired number of points by downsampling or replicating
% point cloud data.
numObservations = size(data,1);
numTransforms = size(data,2);
for i = 1:numObservations    
  for j = 1:numTransforms
    ptCloud = data{i,j}{1};
    %     if ptCloud.Count > numPoints
    percentage = numPoints/ptCloud.Count;
    data{i,j}{1} = pcdownsample(ptCloud,"random",percentage);
  end
%     else    
%         replicationFactor = ceil(numPoints/ptCloud.Count);
%         ind = repmat(1:ptCloud.Count,1,replicationFactor);
%         data{i,j}{1} = select(ptCloud,ind(1:numPoints));
%     end 
end
end