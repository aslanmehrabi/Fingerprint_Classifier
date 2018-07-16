import pickle  # to load and save data
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import dill as pickle


class RunClassifier:

    # receives a classifier(clf) and set of test and train data and corresponding labels. This function apply the classifier on the data and ...
    # ... provide accuracy and related plots of classification. Also it is able to store the trained classifier(with train data) as a pickle file
    def doClassification(self, clf, fVecTrain, fVecTest, labelTrain, labelTest, showPlot=False,savePickleModel=False, clfName='clf', dataType = 'data'):
        print('clf: %s'%(clf) )
        print("Model : %s - Data: %s - SampleSize: %d"%(clfName, dataType,len(labelTrain) + len(labelTest)))
        print('learning started')
        clf.fit(fVecTrain, labelTrain)
        print('model fitted')
        prediction = clf.predict(fVecTest)

        print('# True pred: ', sum(prediction == labelTest))
        print('# False pred: ', sum(prediction != labelTest))

        accuracy = accuracy_score(labelTest, prediction)
        # print("Accuracy :{}", .format(accuracy_score(labelTest, prediction)))
        conf_matrix = confusion_matrix(y_true=labelTest, y_pred=prediction)
        print("Accuracy : ", accuracy)
        print('best parameter: ', clf.best_params_)  # shows the best parametrs of the classifier which was selected by cross validation

        if showPlot:
            plt.matshow(conf_matrix)
            plt.title("Model : %s - Data: %s - SampleSize: %d"%(clfName, dataType,len(labelTrain) + len(labelTest)))
            # plt.show()
            plt.savefig(clfName + ' ' + dataType +' '+str(len(labelTrain) + len(labelTest)) + ' ' + '.png')

        if savePickleModel:  # store pickle file of trained classifier
            storedData = open((str(clf).partition('(')[0]) + '_' + str(len(labelTrain) + len(labelTest)) + '.pckl','wb')
            pickle.dump(clf, storedData)
            storedData.close()

        return prediction, accuracy, conf_matrix, clf.best_params_
