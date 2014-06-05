import numpy as np
from sklearn import preprocessing
import itertools as it

class NormalizedLandmarkDistances:
    """ Base class for features based on normalized peak/neutral landmark coordinate differences """

    FEATURE_NAMES = ['Landmark {0}: normalized distance peak/neutral (x)', 'Landmark {0}: normalized distance peak/neutral (y)']

    def number_of_features(self):
        return 68

    def landmark_number(self, landmark_index):
        assert landmark_index >= 0 and landmark_index < self.number_of_features()

        return landmark_index

    def feature_names(self):
        """ Return a name for each feature, i.e. which landmark number it belongs to and if it is a difference in X or Y """
        return list(y.format(self.landmark_number(x)) for x, y in it.product(range(self.number_of_features()), self.FEATURE_NAMES))

    def __normalize_landmarks(self, landmarks):
        """ Normalize for each face individually """
        for image_sequence_number in range(len(landmarks)):
            landmarks[image_sequence_number, :, :] = preprocessing.scale(landmarks[image_sequence_number, :, :])

        return landmarks

    def normalized_landmark_differences(self, peak_landmarks, neutral_landmarks):
        assert len(peak_landmarks) == len(neutral_landmarks)

        # Normalize landmarks
        neutral_landmarks = self.__normalize_landmarks(neutral_landmarks)
        peak_landmarks = self.__normalize_landmarks(peak_landmarks)

        return (peak_landmarks-neutral_landmarks).reshape((-1, 2*peak_landmarks.shape[1]))

    def extract_features(self, peak_landmarks, neutral_landmarks):
        """ Return the distance between peak and neutral landmarks for x and y direction as features """
        return self.normalized_landmark_differences(peak_landmarks, neutral_landmarks)


    def describe(self):
        return 'Normalized X/Y peak/normal differences for all 68 landmarks'

class SelectedNormalizedLandmarkDistances(NormalizedLandmarkDistances):
    """ Use normalized peak/neutral landmark coordinate differences for certain groups (e.g. left eye, left brow, ...) as features """
    __groups = {
        'left-eye': range(42, 47+1),
        'right-eye': range(36, 41+1),
        'upper-nose': range(27, 30+1),
        'lower-nose': range(31, 35+1),
        'left-brow': range(22, 26+1),
        'right-brow': range(17, 21+1),
        'outer-mouth': range(48, 59+1),
        'inner-mouth': range(60, 67+1)
        # TOOD: add more groups here
        # Get landmark numbers from http://i12r-studfilesrv.informatik.tu-muenchen.de/dmlab2014/images/6/68/Numbered_landmarks.png
    }

    def __init__(self, *group_names):
        for group_name in group_names:
            assert group_name in self.__groups, '{0} is not a valid landmark group'.format(group_name)

        self.group_names = group_names
        self.landmark_numbers = []
        for group_name in self.group_names:
            self.landmark_numbers += self.__groups[group_name]

        self.landmark_numbers = list(set(self.landmark_numbers))

    def landmark_number(self, landmark_index):
        return self.landmark_numbers[landmark_index]

    def number_of_features(self):
        return len(self.landmark_numbers)

    def extract_features(self, peak_landmarks, neutral_landmarks):
        """ Return the distance between selected peak and neutral landmarks for x and y direction as features """

        return self.normalized_landmark_differences(peak_landmarks[:, self.landmark_numbers, :], neutral_landmarks[:, self.landmark_numbers, :])

    def describe(self):
        return 'Normalized X/Y peak/normal differences for landmarks out of {0}'.format(', '.join(self.group_names))
