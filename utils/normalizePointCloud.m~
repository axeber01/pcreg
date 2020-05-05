function dataNorm = normalizePointCloud(data)
% Normalize point cloud by the mean and standard deviation in each dimension

dataNorm = data;
for sample = 1:size(data,1)
  pc = data{sample, 1}{1};
  mu = mean(pc.Location);
  sigma = std(pc.Location);
  for n = 1:size(data, 2) % loop over each rotated and translated point cloud
    pc = data{sample, n}{1};
    dataNorm{sample, n}{1} = pointCloud((pc.Location - mu) ./ sigma);
    
  end
  
end


end

