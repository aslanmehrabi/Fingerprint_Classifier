''''''

# if __name__ == "__main__":


'''@@@@@ better to call a function
calling function rather than importing Mas
'''


from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from ProblemData import ProblemData
from Data_Partition import DataPartition
from RunClassifier import RunClassifier
from sklearn import svm, grid_search, datasets


print('----')

inputFileName = 'E:\\Elm\\others\\CV\\Berlin_start\\Job apply\\Programming Task\Mindoes - Michele\\minodes_recruiting_challenge\\minodes_recruiting_challenge\\data\\fingerprints_gt_ver3.csv'
#inputFileName = 'dataset.csv'
#inputFileName = 'fingerprints_gt_ver3.csv'


readDataUntilRow = -1  # define number of rows which should be read form input file. -1: read the whole data
sampleSize = -1  # define sizse of sample which provides test and train data. -1: readDataUntil will be considered
testPartitionSize = 0.2   # size of the partition of the input size which will be separated as test data

defaultSignalValue = -100  # default(min) value which a sensor(node) can receive
numNodes = 277  # total number of nodes

useStoredData = False  # use stores input data as packle file
storeReadData = False  # store read data from input to a packle file
storeSubSample = False  # store selected subsample as a packle file
useSubSample = False   # load stored subsample from packle file
savePickleModel = False  # store trained classifier after making the learners [new test data can be run on them immediately]

subSamplePcklName = '1000SubSampleVector.pckl'  # name of file to store selected subsample as packle file
storeSubSampleName = 'storeSubSample.pckl'  # name of file to load selected subsample as packle file
storeDataName = 'store.pckl' # name of file to store input file as a packle


def main():

    # loading the data as prblmData
    prblmData = ProblemData(defaultSignalValue=defaultSignalValue, numNodes=numNodes)
    prblmData = prblmData.loadData(useStoredData=useStoredData, inputFileName=inputFileName, storeReadData = storeReadData, storeDataName=storeDataName,
                                   rowReadUntil=readDataUntilRow)

    # partitioning data and providing test and train sets and corresponding labels as dataPar
    dataPar = DataPartition()
    dataPar = dataPar.makeTrainTest(prblmData=prblmData, readSampleSize=sampleSize, testPartitionSize=testPartitionSize,
                                    randomState=0, doNormalize=False, useSubSample=useSubSample, storeSubSample=True,
                                    subSamplePcklName=subSamplePcklName)

    # providing normalized data of dataPar as dataParNormal
    dataParNormal = DataPartition()
    dataParNormal = dataParNormal.makeTrainTest(prblmData=prblmData, readSampleSize=sampleSize,
                                                testPartitionSize=testPartitionSize, randomState=0, doNormalize=True,
                                                useSubSample=useSubSample, storeSubSample=True,
                                                subSamplePcklName=subSamplePcklName)


    # Feature reduction of normalized data by PCA(with pcaNcomponents dimensions) as dataParPca
    pcaNcomponent = 10
    pcaObj = PCA(n_components = pcaNcomponent )
    fit = pcaObj.fit(dataParNormal.fVecTrain)
    dataParPca= DataPartition()
    dataParPca.fVecTrain = pcaObj.fit_transform(dataParNormal.fVecTrain)
    dataParPca.fVecTest = pcaObj.fit_transform(dataParNormal.fVecTest)
    dataParPca.labelTrain = dataParNormal.labelTrain
    dataParPca.labelTest = dataParNormal.labelTest
    dataParPca.isNormalized = True

    # plotting data by first and second principle components of applied PCA on data
    plt.plot(dataParPca.fVecTrain[:,0], dataParPca.fVecTrain[:,1], 'b.')
    plt.title('2D PCA')
    plt.show()


    # Three classifiers (random forest, KNN, SVM) with hyper parameters (will be tuned by cross validation) are defined below:
    # parameter cv shows number of partitions for cross validation of hyper parameter tuning
    # n_jobs = -1 run the data on multiple cores

    # Random Forest:
    # n_estimators: number of trees to make the forest
    #criteria: measuring quality of a split. “gini” for the Gini impurity and “entropy” for the information gain
    # max_features: number of features to consider when looking for the best split
    paramGrid_rf = {'n_estimators': [5, 10, 17, 30], 'criterion': ['gini', 'entropy'],'max_features': ['auto', 0.01, 0.1, 0.9], 'n_jobs': [-1]}
    #paramGrid_rf = {'n_estimators': [30] ,'criterion': ['gini'] ,'n_jobs': [-1]} # n_jobs => runs in parallel
    clf_rf = GridSearchCV(RandomForestClassifier(), paramGrid_rf, cv=3)  # ,scoring='%s_macro' % score


    # KNN:
    # n_neighbors: number of neighbours to consider
    # weights: how to weight labels of neighbors. uniform or distance(consider reverse of the neighbors distance)
    # metric: how to measure the distance: 'minkowski', 'euclidean' or 'manhattan'
    paramGrid_knn = {'n_neighbors': [3, 5, 9, 15], 'weights': ['uniform', 'distance'],'metric': ['minkowski', 'euclidean', 'manhattan']}
    #paramGrid_knn = {'n_neighbors': [5, 9], 'weights': ['distance'],'metric': ['manhattan']}
    clf_knn = GridSearchCV(KNeighborsClassifier(algorithm='kd_tree', n_jobs=-1), paramGrid_knn, cv=3) #, verbose=10  # ,scoring='%s_macro' % score


    # SVM:
    # C : Penalty parameter for miss classification
    # kernel: used kernel 'linear', 'poly', 'rbf' // rbf is more time consuming but seems to be more concordat to problem
    # gamma: kernel coefficient. How far the influence of a single training data reaches [low: far / high: close]
    param_grid_svm = {'C': [0.1], 'kernel': ['rbf'], 'gamma': [0.01]}
    # param_grid_svm = {'C': [0.01, 0.1, 1, 10], 'kernel': ['linear', 'poly', 'rbf'], 'gamma': [0.001, 0.01, 0.1, 1]} # rbf is time consuming comparing to others
    clf_svm = GridSearchCV(svm.SVC(), param_grid_svm, cv=3) # , verbose=10: write the result of each epoc of cv

    clfNames = ['random_forest', 'knn', 'svm']
    dataTypes = ['original ','normalized','nomalized_PCA']

    for idx, clf in enumerate([clf_rf , clf_knn, clf_svm]):
        for idx2,datap in enumerate([dataPar, dataParNormal, dataParPca]):
            runCl = RunClassifier()
            prediction, accuracy, conf_matrix, clf.best_params = runCl.doClassification(clf, datap.fVecTrain,datap.fVecTest, datap.labelTrain,datap.labelTest,showPlot=True,savePickleModel=savePickleModel,clfName = clfNames[idx],dataType = dataTypes[idx2])
            print('\n+++++++++++++++++++\n')


main()


