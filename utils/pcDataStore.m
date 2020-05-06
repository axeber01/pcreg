classdef pcDataStore < matlab.io.Datastore
  
  properties
    MiniBatchSize = 1;
    currIndex = 1;
    nClouds = 2; % number of point clouds to process at once
    nPoints = 10e3; %number of points to sample from the point clouds
    Rstd = 1;
    Tstd = 1;
  end
  
  properties(Dependent)
    Files
  end
  
  properties(Access = private)
    FileDatastore
  end
  
  methods
    function this = pcDataStore(datapath, folds, MiniBatchSize, nClouds, nPoints, Rstd, Tstd)
      
      if nargin == 0
        return;
      end
      
      this.MiniBatchSize = MiniBatchSize;
      this.nClouds = nClouds;
      this.nPoints = nPoints;
      this.Rstd = Rstd;
      this.Tstd = Tstd;
      
      folderList = get_directory_names(datapath);
      
      folderFiles = {};
      for i = folds
        list = dir([datapath, folderList{i}, '/velodyne_points/data/*.bin']);
        currFiles = {};
        for j = 1:length(list)
          currFiles{j} = [list(j).folder, '/', list(j).name];
        end
        
        folderFiles = [folderFiles, currFiles];
        
      end
      
      %fileList = allFiles.name;
      
      %fullFilenames = append(path, fileList);
      this.FileDatastore = fileDatastore(folderFiles,'ReadFcn',@extractTrainingData,'FileExtensions','.bin');
    end
    
    function tf = hasdata(this)
      tf = hasdata(this.FileDatastore);
    end
      
    function [data,info] = read(this)
      
      if ~hasdata(this)
        error('Reached end of data. Reset datastore.');
      end
      
      % Preallocate output.
      batchSize = this.MiniBatchSize;
      data = cell(batchSize, 3);
      info = struct(...
        'Filename',cell(batchSize,1),...
        'FileSize',cell(batchSize,1));
      
      % Read mini-batch size worth of data. The size of data can be
      % less than the specified batch size.
      idx = 0;
      while hasdata(this.FileDatastore)
        idx = idx + 1;
        currData = zeros(this.nPoints, this.nClouds, 3);
        currLabels = zeros(6, this.nClouds - 1);
        [pointDataOrig,info(idx)] = read(this.FileDatastore);
        pcOrig = pointCloud(pointDataOrig);
        % Remove ground plane and ego-vehicle
        pcProcessed = helperProcessPointCloud(pcOrig);
%         sample = randsample(length(pointData), this.nPoints);
%         pointData = pointData(sample, :);
        for n = 1:this.nClouds
          sample = randsample(length(pcProcessed.Location), this.nPoints);
          pointData = pcProcessed.Location(sample, :);
          if n > 1
            % Sample random rotation and translation
            angles = this.Rstd * randn(1, 3);
            translation = this.Tstd * randn(1, 3);
            
            R = eul2rotm(angles);
          else
            angles = [0, 0, 0];
            translation = [0, 0, 0];
            
            R = eul2rotm(angles);
          end
          
          tform = rigid3d(R, translation);
          label = [angles / this.Rstd, translation / this.Tstd];
          pc = pctransform(pointCloud(pointData), tform);
          
          if n == 1
            mu = 0; %mean(pc.Location);
            sigma = std(pc.Location(:));
          else
            currLabels(:, n - 1) = label;
          end
          
          pc = pointCloud((pc.Location - mu) / sigma);
          
          
          currData(:, n, :) = pc.Location;
        end
        data(idx, :) = {currData, currLabels, sigma};
        
        if idx == batchSize
          break;
        end
      end
      
      data = data(1:idx,:);
      info = info(1:idx);
      
    end
    
    function reset(this)
      reset(this.FileDatastore);
    end
    
    function files = get.Files(this)
      files = this.FileDatastore.Files;
    end
    
    function set.Files(this,files)
      this.FileDatastore.Files = files;
    end
    
  end

  methods(Access=protected)
    function newds = copyElement(this)
      newds = pcDataStore();
      newds.FileDatastore = copy(this.FileDatastore);
      newds.MiniBatchSize = this.MiniBatchSize;
      newds.nClouds = this.nClouds;
      newds.nPoints = this.nPoints;
      newds.Rstd = this.Rstd;
      newds.Tstd = this.Tstd;
    end
  end
    
end

function dataOut = extractTrainingData(fname)

dataOut = readVelodyne(fname);
dataOut = dataOut(:, 1:3);

end

function velo = readVelodyne(frame)
% load velodyne points
fid = fopen(frame,'rb');
velo = fread(fid,[4 inf],'single')';
fclose(fid);
end
