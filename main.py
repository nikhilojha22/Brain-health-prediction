
import numpy as np
import nibabel as nib
import importlib
import time
import random
import sys
sys.path.append('../src')
import imageAnalysis as ia
import MachineLearningAlgorithms as ml
import matplotlib.pyplot as plt
import printh as pt

path = "./data/"

print("\nLibraries imported")



    
ia = importlib.reload(ia)
ml = importlib.reload(ml)

print("\nLibraries reloaded")


def loadData( path, strDataset, strName, nSamples ):
    
   
    xSize = 176
    ySize = 208
    zSize = 176

    xLimMin = 14
    xLimMax = 18
    yLimMin = 12
    yLimMax = 15
    zLimMin = 3
    zLimMax = 20

    
    datasetDic = {}

    for i in range(nSamples):
      
        imageName = strName + str(i + 1)
        imagePath = path + "/" + strDataset + "/" + imageName + ".nii"
        
     
        imageRaw = nib.load(imagePath)
        if(i==0):
         print(imageRaw)
       
        datasetDic[i] = imageRaw.get_data()[xLimMin:xSize-xLimMax, \
        yLimMin:ySize-yLimMax, zLimMin:zSize-zLimMax, 0]
        
       
      
    return datasetDic


def featuresExtraction(datasetDic, featureDic):
    
   
    
    dataSet = ml.Features(datasetDic)
    #print ("nikhil")
           
    featureMatrix, binEdges = dataSet.featureExtraction(**featureDic)
    #print("nikhil")
    
    nFeatures = featureMatrix.shape[1]
    
    
    return featureMatrix, binEdges

print("\nFunctions loaded")

#################################################################################################################################################

strLabel = "targets.csv"


label = np.genfromtxt(path+strLabel, delimiter=',').astype(int)


nSamples = label.size
    
print("\nLabels loaded. There are " + str(nSamples) + " samples in the dataset")


strDataset = "set_train"


strName = "train_"



datasetDic = loadData( path, strDataset, strName, nSamples )


imageCroped = ia.ImageProperties(datasetDic[200])

imageCroped.toPlot()


print("\nThe dataset dictionary containing all the 3D images of the labeled \
dataset has been created")        



    
ml = importlib.reload(ml)

featureDic ={}

featureDic["gridOperation"] = { "nGrid":(10,10,10), "npoly":1, \
                                "typeOp":["histogram"], "binEdges":10}

            


featureMatrix, binEdges = featuresExtraction( datasetDic, featureDic)
featurevalue=pt.printm
#featurevalue.printmatrix()  
ml = importlib.reload(ml)

data2Predict = ml.Prediction(featureMatrix, label)    

temp= True

if temp== True:
  
    methodML = "AdaBoost"
    adaBoostDic={"n_estimators":150, "learning_rate":0.005}
    
    methodML = "Bagging"
    baggingDic={"n_estimators":100, "n_jobs":-1, "bootstrap_features": False}
    
    methodML = "Gradient Boosting"
    gradBoostDic={"n_estimators":100, "learning_rate":0.05, "max_depth":1, "loss": "deviance"}
    
    methodML = "Random Forest"
    rdmForrestDic={"n_estimators":300, "criterion":"gini", "class_weight":None, \
                         "bootstrap":True, "oob_score":True, "n_jobs":-1}
                         
    methodML = "SVM"
    svmDic= {"C": 0.1, "kernel":"poly", "degree":1, "probability":True}
    
    methodDic =[svmDic, rdmForrestDic, adaBoostDic, baggingDic, gradBoostDic]
    methodML = ["SVM", "Random Forest", "AdaBoost", "Bagging", "Gradient Boosting"]
    
   
    methodDic =[rdmForrestDic]
    methodML = ["Random Forest"]
   
    methodDic =[svmDic]
    methodML = ["SVM"]
   
    methodDic =[baggingDic]
    methodML = ["Bagging"]
   
    methodDic =[gradBoostDic]
    methodML = ["Gradient Boosting"]
   
    methodDic =[adaBoostDic]
    methodML = ["AdaBoost"]
    
    methodDic =[svmDic, rdmForrestDic, adaBoostDic, baggingDic, gradBoostDic]
    methodML = ["SVM", "Random Forest", "AdaBoost", "Bagging", "Gradient Boosting"]
   
    methodDic =[svmDic]
    methodML = ["SVM"]
  
    methodDic =[baggingDic]
    methodML = ["Bagging"]
    
    methodDic =[rdmForrestDic]
    methodML = ["Random Forest"]
 
    nMethods = len(methodML)
   
    print("")
    print("********************The method to train our model is random forest ************************")
    print("")
    
   
    score = data2Predict.crossValidation(methodDic, nFold=10,  \
                                  typeCV="random", methodList=methodML, \
                                  stepSize = 0.01)
        
      
   
    
   
    ml = importlib.reload(ml)
    ensembleSelectionChosen = True
    
   
    data2Predict = ml.Prediction(featureMatrix, label)  
    
    methodDic =[svmDic, rdmForrestDic, adaBoostDic,gradBoostDic]
    methodML = ["SVM", "Random Forest", "AdaBoost", "Gradient Boosting"]
    
    classifierList, weightModel, score = data2Predict.ensembleSelection(methodDic,\
                                  Ratio=0.7, typeDataset="random", methodList=methodML,\
                                  stepSize=0.001)

      
ml = importlib.reload(ml)

strDataset = "set_test"


strName = "test_"


datasetTestDic = loadData( path, strDataset, strName, 138 )
nSampleTest = len(datasetTestDic)




for k,v in featureDic.items():
    if ("subhistogram" in featureDic[k]["typeOp"]) or ("histogram" in featureDic[k]["typeOp"]):
        featureDic[k]["binEdges"] = binEdges


featureMatrixTest, _ = featuresExtraction(datasetTestDic, featureDic)


ml = importlib.reload(ml)
    
        
unlabeledData= ml.Prediction(featureMatrixTest)



methodML=["Random Forest"]
#methodML = ["SVM"]
if temp == True:
    testPrediction = np.zeros([nSampleTest])
    
    for i in range(nMethods): 
        predictionModel = unlabeledData.predict(featureMatrixTest, method=methodML[i], \
                                      labelValidation = [], classifier=classifierList[i])
        testPrediction += weightModel[i]*predictionModel
        testPrediction=predictionModel
    #print(predictionModel)
else:
    testPredictionArray = []
    for i in range(len(classifiersArray)):
        testPredictionArray.append(unlabeledData.predict(method=0, features=featureMatrixTest, classifier=classifiersArray[i]))


print("\n The prediction for the non-labeled dataset")    


if temp == True:
    date = (time.strftime("%d-%m-%Y %Hh%Mm%S"))
    methods = ""
    
    for i in range(nMethods):
        methods += methodML[i] + " "
    
    fileStr = methods + date + ".csv"

    fileIO = open( path + fileStr,'w' )
    fileIO.write( 'ID,Prediction\n' )
    answer = testPrediction
    for i in range( len( answer ) ):
        fileIO.write( str(i+1) + ',' + str(answer[i]).strip('[]') + '\n' )
    fileIO.close()
else:    
    for i in range(len(classifiersArray)):
        fileStr = classifiersType[i]+ ".csv"
    
        fileIO = open( path + fileStr,'w' )
        fileIO.write( 'ID,Prediction\n' )
        answer = testPredictionArray[i]
        for i in range( len( answer ) ):
            fileIO.write( str(i+1) + ',' + str(answer[i]).strip('[]') + '\n' )
        fileIO.close()

#print("\n The prediction has been written in a .csv file") 
#strLabel="targets.csv"
cnth=0
arrsize=testPrediction.size
'''
for i in testPrediction:
  if(testPrediction[i]>0.65):
     cnth= chth +1
'''
chth = cnth/float(arrsize)
labelHealth = np.zeros(nSamples)
label_i=label
arrsum=np.sum(testPrediction)
healthyProportion = round(100 *cnth, 2)
print("healthy percentage:-",end="")
print(round(100*109/138,2))
print("sick percentage:-",end="")
print(round(100*29/138,2)) 

plt.show()  
