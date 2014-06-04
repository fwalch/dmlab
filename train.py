#!/usr/bin/env python3

import argparse
from train.json_loader import JsonLoader
from train.landmark_features import *
from sklearn import cross_validation, metrics, tree, ensemble
import itertools as it

class Training:
    __FOLDS = 10

    def __init__(self, feature_extractors, classifier_generator, training_repetitions):
        self.feature_extractors = feature_extractors
        self.classifier_generator = classifier_generator
        self.training_repetitions = training_repetitions

    def train(self):
        neutral_landmarks, peak_landmarks, emotions = JsonLoader.load_default()

        # Try with different sets of extracted features
        for feature_extractor in self.feature_extractors:

            # Extract features
            feature_names = feature_extractor.feature_names()
            features = feature_extractor.extract_features(peak_landmarks, neutral_landmarks)

            assert len(features) == len(emotions)

            scores = []
            confusion_matrices = []
            classifiers = []

            # Repeat training N times
            for _ in range(self.training_repetitions):
                folds = cross_validation.StratifiedKFold(features, n_folds=self.__FOLDS)

                for train_selector, test_selector in folds:

                    # Split into training/test set
                    X_train = features[train_selector]
                    X_test = features[test_selector]
                    y_train = emotions[train_selector]
                    y_test = emotions[test_selector]

                    # Create classifier (decision tree, ...)
                    classifier = self.classifier_generator()
                    classifier = classifier.fit(X_train, y_train)

                    # Evaluate
                    score = classifier.score(X_test, y_test)
                    confusion_matrix = metrics.confusion_matrix(y_test, classifier.predict(X_test))

                    scores.append(score)
                    confusion_matrices.append(confusion_matrix)
                    classifiers.append(classifier)

            # Output results
            print()
            print('#', feature_extractor.describe())
            print('  Generated {classifiers} classifiers ({folds} folds, {repetitions} repetitions) of type {classifier_type}'.format(classifiers=self.training_repetitions*self.__FOLDS, repetitions=self.training_repetitions, classifier_type=type(self.classifier_generator()), folds=self.__FOLDS))
            print('  Prediction based on {0} features for {1} image sequences'.format(features.shape[1], features.shape[0]))
            print('  Average accuracy over all folds and repetitions:', sum(scores)/len(scores))
            print('  Best accuracy:', max(scores))
            print('  Best confusion matrix:')
            print(confusion_matrices[scores.index(max(scores))])

            # Export decision trees; create PNG with ./dot_to_png.sh
            if type(self.classifier_generator()) == type(tree.DecisionTreeClassifier):
                best_classifier = classifiers[scores.index(max(scores))]
                feature_descriptions = list('Feature {0}, {1}'.format(x, y) for x, y in it.product(range(int(features.shape[1]/2)), feature_names))

                with open("{0}.dot".format(feature_extractor.__name__), 'w') as outfile:
                    outfile = tree.export_graphviz(best_classifier, out_file=outfile, feature_names=feature_descriptions)

if __name__ == '__main__':
    # Classifier creation functions, used later
    def create_random_forest():
        return ensemble.RandomForestClassifier(n_estimators=50)

    def create_decision_tree():
        return tree.DecisionTreeClassifier()

    # Parse command line options
    parser = argparse.ArgumentParser(description='Train landmark-based classifiers for facial expression recognition')
    parser.add_argument('--landmark-groups', action='append', nargs='+')
    parser.add_argument('--individual-landmarks', action='store_true', default=True)
    parser.add_argument('--repetitions', type=int, default=1, required=False)
    parser.add_argument('--classifier', choices=['random-forest', 'decision-tree'], default='random-forest')

    args = parser.parse_args()

    # Decide on classifier
    classifier_generator = create_random_forest if args.classifier == 'random-forest' else create_decision_tree

    # Add different feature sets to use
    feature_extractors = []
    if args.landmark_groups:
        for landmark_group in args.landmark_groups:
            feature_extractors.append(SelectedNormalizedLandmarkDistances(*landmark_group))

    if args.individual_landmarks:
        feature_extractors.append(NormalizedLandmarkDistances())

    # Start training
    Training(feature_extractors, classifier_generator, args.repetitions).train()