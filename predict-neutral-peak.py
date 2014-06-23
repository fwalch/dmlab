#!/usr/bin/python3

import argparse
import os
import pickle
import warnings
import itertools as it
from sklearn.preprocessing import scale
import numpy as np

CLASSIFIER_FILE = 'classifier-neutral-peak.pkl'

def predict_emotion(neutral_landmarks, peak_landmarks):
    classifier_path = os.path.join('..', '..', CLASSIFIER_FILE)
    assert os.path.exists(classifier_path), 'Classifier not saved; run {0} with the --prepare option'.format(__file__)

    with open(classifier_path, 'rb') as infile:
        classifier = pickle.load(infile)

    neutral_landmarks = neutral_landmarks.reshape((1, -1, 2))
    peak_landmarks = peak_landmarks.reshape((1, -1, 2))
    data = get_features(neutral_landmarks, peak_landmarks, select=False)
    prediction = classifier.predict(data)
    return {
        0: 'neutral',
        1: 'anger',
        2: 'contempt',
        3: 'disgust',
        4: 'fear',
        5: 'happiness',
        6: 'sadness',
        7: 'surprise'
    }[int(prediction)]

def __normalize_landmarks(landmarks):
    """ Normalize for each face individually """
    for image_sequence_number in range(len(landmarks)):
        landmarks[image_sequence_number, :, :] = scale(landmarks[image_sequence_number, :, :])

    return landmarks

def get_features(neutral_landmarks, peak_landmarks, *, select=True):
    assert neutral_landmarks.shape == peak_landmarks.shape

    norm_neutral_landmarks = __normalize_landmarks(neutral_landmarks)
    norm_peak_landmarks = __normalize_landmarks(peak_landmarks)

    selector = [17,21,22,26,31,32,33,34,35,36,39,42,45,48,49,50,51,52,53,54,55,56,57,58,59,61,62,63,65,66,67,18,20,23,25,0,8,16]
    if select:
        assert neutral_landmarks.shape[1] == 68, 'Wrong number of input landmarks'
        features = (norm_peak_landmarks-norm_neutral_landmarks)[:, selector, :]
    else:
        assert neutral_landmarks.shape[1] == len(selector), 'Wrong number of input landmarks'
        features = (norm_peak_landmarks-norm_neutral_landmarks)

    return features.reshape((len(norm_neutral_landmarks), 2*len(selector)))

if __name__ == '__main__':
    from train.json_loader import JsonLoader
    from sklearn.pipeline import Pipeline, FeatureUnion
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
    from sklearn.svm import LinearSVC
    from sklearn.lda import LDA
    from sklearn.grid_search import GridSearchCV
    from sklearn.decomposition import PCA
    from sklearn.feature_selection import SelectKBest, RFECV
    from sklearn.svm import SVC
    from sklearn.naive_bayes import GaussianNB

    parser = argparse.ArgumentParser(description='Create pipeline to use with the demo.')
    parser.add_argument('--prepare', action='store_true', default=False, required=True)
    args = parser.parse_args()

    if args.prepare:
        feature_extractor = FeatureUnion([
            ('pca', PCA()),
            ('univ_select', SelectKBest())
        ])
        pipeline = Pipeline([
            #('features', LDA()),
            ('predict', RandomForestClassifier())
        ])
        parameters = dict(
            #features__pca__n_components=[1,2,3,4,5],
            #features__n_components=[1,2,3,4,5],
            #features__univ_select__k=list(range(1, 68, 2)),
            #features__k=list(range(1, 68, 2)),
            #features__n_estimators=[10,20,50,100,200],
            predict__n_estimators=[50,200]
        )

        # Grid search with 10 folds
        grid_search = GridSearchCV(pipeline, parameters, verbose=1, cv=10)

        neutral_landmarks, peak_landmarks, emotions = JsonLoader.load_without(None)
        assert len(neutral_landmarks) == len(emotions)
        data = get_features(neutral_landmarks, peak_landmarks)
        target = emotions

        # Suppress scipy deprecation warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            grid_search.fit(data, target)

        print("Best score: %0.3f" % grid_search.best_score_)
        print("Best parameters set:")
        best_parameters = grid_search.best_estimator_.get_params()
        for param_name in sorted(parameters.keys()):
            print("\t%s: %r" % (param_name, best_parameters[param_name]))

        with open(CLASSIFIER_FILE, 'wb') as outfile:
            pickle.dump(grid_search.best_estimator_, outfile)
