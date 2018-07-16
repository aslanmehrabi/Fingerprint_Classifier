import dill as pickle
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,normalize
import copy

class DataPartition:

    fVecTrain = [] # feature vector for train set
    fVecTest = [] # feature vector for test set
    labelTrain = [] # labels of train set
    labelTest = [] # labels of test set
    isNormalized = False  # shows whether the data has been normalized


# split data to train and test partitions of specific size defined by user + load stores feature vectors(if asked by user) + do data normalization (if asked by user)
    def makeTrainTest (self, prblmData, readSampleSize = -1, testPartitionSize = 0.3,randomState = 0 ,doNormalize = False, useSubSample = False, storeSubSample = False, subSamplePcklName = ''):
        if useSubSample:
            storedData = open(subSamplePcklName, 'rb')
            self = pickle.load(storedData)
            storedData.close()
            return self

        else :
            if(readSampleSize == -1):
                readSampleSize = prblmData.numRows
            sampleSize = min(readSampleSize, prblmData.numRows)  # select sampleSize of random indices of rows


            randInd = random.sample(range(prblmData.numRows), sampleSize)
            self.isNormalized = doNormalize
            if doNormalize :
                normalSigMat = StandardScaler().fit_transform(prblmData.sigMat)
                self.fVecTrain, self.fVecTest, self.labelTrain, self.labelTest = train_test_split(normalSigMat[randInd],prblmData.zone[randInd],test_size=testPartitionSize,random_state=randomState)

            else:
                self.fVecTrain, self.fVecTest, self.labelTrain, self.labelTest = train_test_split(prblmData.sigMat[randInd],prblmData.zone[randInd],test_size=testPartitionSize,random_state=randomState)

            if storeSubSample:  # stores train and test data as pickle file
                storedData = open(subSamplePcklName, 'wb')
                pickle.dump(self, storedData)
                storedData.close()

        return self

