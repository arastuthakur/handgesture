%code for training.
clc
clear all
close all
warning off
g=alexnet; %for image to be processed.
layers=g.Layers;
layers(23)=fullyConnectedLayer(7);
layers(25)=classificationLayer;
allImages=imageDatastore('Hand Dataset','IncludeSubfolders',true, 'LabelSource','foldernames'); %My initial Data Set
opts=trainingOptions('sgdm','InitialLearnRate',0.001,'MaxEpochs',20,'MiniBatchSize',64);
myNet1=trainNetwork(allImages,layers,opts);
save myNet1; %folder name to save the data.