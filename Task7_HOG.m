%% Multi-classification of images with HOG features, svm training, 1 vs 1  

%% 1 Dataset, including training and test datasets

% Path of the given image dataset
path=fullfile('E:\Matlab_doc\robot_v\homework1\p_dataset_26\');
imds=imageDatastore(path,'IncludeSubfolders',true,'LabelSource','foldernames');
[imdsTrain, imdsTest] = splitEachLabel(imds, 0.75, 'randomized');

%% Show the type of images trained Labels and the number Count
Train_disp = countEachLabel(imdsTrain);
disp(Train_disp);
  
%% 2 HOG feature extraction for each image in the training set, same for the test image  

% Pre-processing the image, mainly to obtain the feature size, 
% which is related to the image size and the Hog feature parameters  
imageSize = [128,128]; 
image1 = readimage(imdsTrain,1);  
scaleImage = imresize(image1,imageSize);  

% Feature extraction
[features, visualization] = extractHOGFeatures(scaleImage);   
  
% Feature extraction for all training images  
numImages = length(imdsTrain.Files);  
featuresTrain = zeros(numImages,size(features,2),'single');  
for i = 1:numImages  
    imageTrain = readimage(imdsTrain,i);  
    imageTrain = imresize(imageTrain,imageSize);  
    featuresTrain(i,:) = extractHOGFeatures(imageTrain);  
end  
  
% All training image labels  
trainLabels = imdsTrain.Labels;  
  
% Start svm multiclassification training 
% Note: fitcsvm for binary classification, fitcecoc for multiclassification,1 VS 1 method  
classifer = fitcecoc(featuresTrain,trainLabels);  

%% Feature extraction for all test images 

numImagesTest = length(imdsTest.Files);  
featuresTest = zeros(numImagesTest,size(features,2),'single');  
for i = 1:numImagesTest
    imageTest = readimage(imdsTest,i);  
    imageTest = imresize(imageTest,imageSize);  
    featuresTest(i,:) = extractHOGFeatures(imageTest);  
end  

%% Accuracy

YTest = imdsTest.Labels; 
YPred = predict(classifer,featuresTest);
accuracy = mean(YPred == YTest)

%% 3 Identify HD44780A00

% The path to the image to be recognized
% Noting that the image needs to be in the folder corresponding to the label
path=fullfile('E:\Matlab\toolbox\libsvm-3.31\libsvm-3.31\matlab\task2\');
imdsDo=imageDatastore(path,'IncludeSubfolders',true,'LabelSource','foldernames');

% Pre-processing of task images, including corrosive expansion, etc.
imdsDo.ReadFcn = @(x)preprocessImage(x);
numDo = length(imdsDo.Files);  

figure;
for i = 1:numDo
    testImage = readimage(imdsDo,i);  
    scaleTestImage = imresize(testImage,imageSize);  
    featureTest = extractHOGFeatures(scaleTestImage); 
    predictIndex= predict(classifer,featureTest);  
    subplot(2,5,i);
    imshow(testImage);
    % figure;imshow(testImage);  
    title(['predictImage: ',char(predictIndex)]);  
end  

%% Image processing functions
function I = preprocessImage(filename)
    I = imread(filename);
    I = rgb2gray(I);
    % Corrosion
    se = strel('disk', 6); % Corrosion radius of 5
    I = imerode(I, se);
    % Inflation
    I = imdilate(I, se); 
    pad_m = 28;
    pad_n = 15;
    % Use the padarray function to add a white border around an image
    % The fill pixel value is 255, indicating white
    I = padarray(I, [pad_m, pad_n], 255, 'both');
    I = imresize(I, [128, 128]);
    I = repmat(I, [1, 1, 1]);
end