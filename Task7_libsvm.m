%% Training methods using libsvm

%% 1 Dataset, including training and test datasets

% Path of the given image dataset
path=fullfile('E:\Matlab_doc\robot_v\homework1\p_dataset_26\');
imds=imageDatastore(path,'IncludeSubfolders',true,'LabelSource','foldernames');
[imdsTrain, imdsTest] = splitEachLabel(imds, 0.75, 'randomized');

%% 2 Labelling of image datasets and formatting of images

trainLabels = imdsTrain.Labels;  
testLabels = imdsTest.Labels;  
% Need to convert labels to double format to be recognized by svmtrain
trainLabels = double(trainLabels);
testLabels = double(testLabels);

% After transpose, each row is transformed into a matrix of number*pixel
numImages = length(imdsTrain.Files);  
pixel_num = 128*128;
trainImages = zeros(numImages,pixel_num,'single'); 
for i = 1:numImages  
    imageTrain = readimage(imdsTrain,i);  
    imageTrain = reshape(imageTrain, 1, []);
    trainImages(i,:) = imageTrain;
end  
 
numImages2 = length(imdsTest.Files);  
testImages = zeros(numImages2,pixel_num,'single'); 
for i = 1:numImages2  
    imageTest = readimage(imdsTest,i);  
    imageTest = reshape(imageTest, 1, []);
    testImages(i,:) = imageTest;
end  
% Image conversion to double format
trainImages = double(trainImages);
testImages = double(testImages);

%% 3 SVM training

% linear SVM
tic; % Timing
options='-s 0 -t 0 -q'; % Selecting training parameters
model=svmtrain(trainLabels, trainImages, options); 
% Correctness evaluation using the svmpredict function
[~,accuracy,~]=svmpredict(testLabels, testImages, model);
fprintf('Time: %0.2f s. \n', toc); 

% Show accuracy
accuracy

%% 4 Identify HD44780A00

% The path to the image to be recognized
% Noting that the image needs to be in the folder corresponding to the label
path=fullfile('E:\Matlab\toolbox\libsvm-3.31\libsvm-3.31\matlab\task\');
imdsDo=imageDatastore(path,'IncludeSubfolders',true,'LabelSource','foldernames');

% Pre-processing of task images, including corrosive expansion, etc.
imdsDo.ReadFcn = @(x)preprocessImage(x);

numDo = length(imdsDo.Files); 
ownImages = zeros(numDo,pixel_num,'single'); 
for i = 1:numDo 
    imageDo = readimage(imdsDo,i);  
    imageDo = imresize(imageDo,[128,128]);
    imageDo = reshape(imageDo, 1, []);
    ownImages(i,:) = imageDo;
end  

Label = imdsDo.Labels;
ownImages = double(ownImages);
[TaLabel,Relist]=grp2idx(Label);% CNN labels can be letters, SVMs must be numbers
[Task_predict,~,~]=svmpredict(TaLabel,ownImages,model);

for i=1:10
    % Converts the validation tag back to its original form
    TaLabel1(i)=Relist(TaLabel(i));
    % Converts the predicted label back to its original form
    Task_predict1(i)=Relist(Task_predict(i));
end

% Converting validation labels to categorical variables
TaLabel1=categorical(cellstr(TaLabel1));
% Converting predictive labels to categorical variables
Task_predict1=categorical(cellstr(Task_predict1));

% Calculate prediction accuracy, retaining 4 valid digits
precision_v=double(vpa(sum(Task_predict1==TaLabel1)/numel(Task_predict1),4));

figure
confusionchart(TaLabel1,Task_predict1)

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
    I = padarray(I, [pad_m, pad_n], 0, 'both');
    I = imresize(I, [128, 128]);
    I = repmat(I, [1, 1, 1]);
end
