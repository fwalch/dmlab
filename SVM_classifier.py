__author__ = 'paufabregatpappaterra'

from sklearn.svm import SVC
from sklearn import cross_validation, metrics

import numpy as np

def classifier(X, Y):
    """
    def: Train the SVM classifier and test the performance with cross-validation 5-fold
    @param X: training set
    @param Y: class labels
    """

    #SVM classifier
    clf = SVC(probability=True)

    #Stratified folds: each set contains the same percentage of samples of each target class as the complete set.
    cv = cross_validation.StratifiedKFold(Y, n_folds=4)

    #predict(): Predict target values of X given a model (low-level method)
    #predict_proba(): Predict probabilities
    scoreAVG = []
    confusionadd = np.zeros((7,7))
    Ytest_a, pred_a = [], []


    #Cross Validation 7-fold
    for traincv, testcv in cv:


        #Perform prediction
        pred = clf.fit(X[traincv], Y[traincv]).predict(X[testcv])


        #Accuracy & Confusion matrix metrics
        score = metrics.accuracy_score(Y[testcv], pred, normalize=True) #Normalize= False: returns numeber of assertions
                                                                        #Normalize= True: returns % correctly classified

        confusion = metrics.confusion_matrix(Y[testcv], pred, labels=None)


        scoreAVG.append(score)
        confusionadd = confusionadd + confusion
        Ytest_a.extend(Y[testcv])
        pred_a.extend(pred)

    scoreAVG = np.array(scoreAVG)
    report = metrics.classification_report(Ytest_a, pred_a, labels = [1,2,3,4,5,6,7], target_names=['Anger', 'Contempt', 'Disgust', 'Fear', 'Happy','Sadness','Surprise'])

    print("AVG accuracy-score:   %0.3f, %.3f, %.3f\n" % (scoreAVG.mean(), scoreAVG.std(), np.median(scoreAVG)))
    print("AVG confusion-matrix:\n%s\n" % confusionadd)
    print("AVG report\n%s" %  report)


def main():

    dX = np.load(open('datasetX.npy', 'rb'))
    dY = np.load(open('datasetY.npy', 'rb'))
    classifier(dX,dY)



if __name__ == '__main__':
    main()
