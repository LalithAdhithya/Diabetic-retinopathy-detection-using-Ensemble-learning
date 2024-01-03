# Diabetic-retinopathy-detection-using-Ensemble-learning
An Ensemble model of ResNet-50,DenseNet121 and VGC16 
This code aims to make an ensemble model of DenseNet-121,InceptionV3 and ResNet-50 to detect Diabetic Retinopathy 

The data is initially Loaded,augemntation and normalize is performed
The augumented data is loaded in the pretrained DenseNet-121 model first and the results are evaluated
similarly the data is loaded to the remaining two models and the output is saved

Now we concate the results of all the 3 models and give that as the input to a new ensemble model which genereates a better Result that combines results of all the three models


