numTrainFiles=191;
numTestFiles=63;
% Set the path to the dataset
path=fullfile('D:\Study\5411\p_dataset_26\p_dataset_26\');
imds=imageDatastore(path,'IncludeSubfolders',true,'LabelSource','foldernames');
% Divide the data set randomly in the ratio 3:1
[imdsTrain,imdsTest] = splitEachLabel(imds,0.75,'randomize');
% Calculate the number of samples for each label in the training and test sets
trainSetDetial=countEachLabel(imdsTrain);
testSetDetial=countEachLabel(imdsTest);

% Set the size of the input image
inputSize = [128 128 1];

img=readimage(imdsTrain,1);
size(img)

layers=[
    imageInputLayer([128 128 1])

    convolution2dLayer(5,16,'Padding','same')
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2,"Stride",2)

    convolution2dLayer(5,32,'Padding','same')
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2,"Stride",2)

    convolution2dLayer(5,64,'Padding','same')
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2,"Stride",2)

    convolution2dLayer(5,128,'Padding','same')
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2,"Stride",2)

    convolution2dLayer(5,256,'Padding','same')
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2,"Stride",2)

    convolution2dLayer(5,512,'Padding','same')
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2,"Stride",2)
    
    dropoutLayer

    fullyConnectedLayer(7)
    softmaxLayer
    classificationLayer]; 


options=trainingOptions('adam',...
    'InitialLearnRate',0.001,...
    'MaxEpochs',20,...
    'Shuffle','every-epoch',...
    'ValidationData',imdsTest,...
    'ValidationFrequency',20,... 
    'MiniBatchSize',64,...
    'Verbose',true,...
    'LearnRateSchedule','piecewise',...
    'ExecutionEnvironment','gpu',...
    'Plots','training-progress');

% Images augmenter
imageAugmenter = imageDataAugmenter( ...
    'RandRotation',[-45,45], ...
    'RandScale', [1.2 1.2], ...
    'RandXTranslation',[-5,5], ...
    'RandYTranslation',[-5,5]);
imdsAug = augmentedImageDatastore(inputSize,imdsTrain,'DataAugmentation',imageAugmenter); 

% Network definition and training
tic;
net=trainNetwork(imdsTrain,layers,options);
% net=trainNetwork(imdsAug,layers,options);
fprintf('Training Time: %0.2f s. \n', toc); 

% Network classification prediction
% Save the prediction results in the variable YPred
YPred=classify(net,imdsTest);
YTest=imdsTest.Labels;
% Calculate prediction accuracy
accuracy=sum(YPred==YTest)/numel(YTest)

% Visualize the confusion matrix
figure
confusionchart(YTest,YPred)

% Task Predict
taskpath=fullfile('D:\Study\5411\task(1)\');
imdsTask=imageDatastore(taskpath,'IncludeSubfolders',true,'LabelSource','foldernames');
img=readimage(imdsTask,1);
size(img)
% Perform image pre-processing
imdsTask.ReadFcn = @(x)preprocessImage(x);


% Use the classify function to predict the classification of these images
% Save the prediction results in the variable YTaskpredict
YTaskpredict=classify(net,imdsTask);
YTrue=imdsTask.Labels;
% Calculate prediction accuracy
accuracy=sum(YTaskpredict==YTrue)/numel(YTrue)

% Calculate the confusion matrix and visualize the confusion matrix
figure
confusionchart(YTrue,YTaskpredict)


% Define image pre-processing functions
function I = preprocessImage(filename)
    I = imread(filename);
    I = rgb2gray(I);
    % Corrosion
    se = strel('disk', 6); % Corrosion radius of 6
    I = imerode(I, se);
    % Inflation
    se2=strel('disk',6); % Corrosion radius of 6
    I = imdilate(I, se2); 
    I = imresize(I, [128, 128]);
    I = repmat(I, [1, 1, 1]);
%     figure()
%     imshow(I)
end