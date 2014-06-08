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

class SymmetricAndNormalizedLandmarkDistances(NormalizedLandmarkDistances):
    __vertical = [(0, 16), (1,15), (2,14),(3,13), (4,12),(5,11),(6,10),(7,9), (48,54),(60,64),(59,55),(58,56),(67,65),(61,63),(49,53),(50,52),(31,35),(32,34),(36,39),(42,45),(36,45),(39,42),(17,20),(22,26),(17,26),(21,22), (48,60),(64,54)]
    __horizontal = [(51,57),(62,66),(50,58),(61,67),(53,55),(52,56),(63,65),(49,59),(37,41),(38,40),(43,47),(44,46),(24,22),(24,26),(19,17),(19,21),(33,8),(30,8),(39,8),(42,8),(42,27),(39,27)]

    def number_of_features(self):
        return super().number_of_features() + len(self.__vertical) + len(self.__horizontal)

    def landmark_number(self, landmark_index):
        assert landmark_index >= 0 and landmark_index < self.number_of_features()

        if landmark_index < 68:
            return super().landmark_number(landmark_index)

        if landmark_index-68 < len(self.__vertical):
            return '{0}/{1}'.format(self.__vertical[landmark_index-68][0], self.__vertical[landmark_index-68][1])

        return '{0}/{1}'.format(self.__horizontal[landmark_index-68-len(self.__vertical)][0], self.__horizontal[landmark_index-68-len(self.__vertical)][1])

    def describe(self):
        return super().describe() + ', plus symmetric differences for selected landmarks'

    def normalized_symmetric_landmark_differences(self, peak_landmarks, neutral_landmarks):
        assert len(peak_landmarks) == len(neutral_landmarks)

        lm_one = [one for (one,two) in self.__vertical+self.__horizontal]
        lm_two = [two for (one,two) in self.__vertical+self.__horizontal]

        peak_diffs = self.normalized_landmark_differences(peak_landmarks[:, lm_one, :], peak_landmarks[:, lm_two, :])
        neutral_diffs = self.normalized_landmark_differences(neutral_landmarks[:, lm_one, :], neutral_landmarks[:, lm_two, :])

        return (np.abs(peak_diffs-neutral_diffs)).reshape((-1, 2*len(lm_one)))

    def extract_features(self, peak_landmarks, neutral_landmarks):
        default_features = self.normalized_landmark_differences(peak_landmarks, neutral_landmarks)

        symmetric_features = self.normalized_symmetric_landmark_differences(peak_landmarks, neutral_landmarks)

        return np.concatenate((default_features, symmetric_features), axis=1)

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
        'inner-mouth': range(60, 67+1),
        'reduced': [17,19,24,26,37,41,44,46,31,32,34,35]
        # TOOD: add more groups here
        # Get landmark numbers from http://i12r-studfilesrv.informatik.tu-muenchen.de/dmlab2014/images/6/68/Numbered_landmarks.png
    }
    __diff_groups = {
        'reduced': [(21,22),(51,57),(48,54)]
    }

    def __init__(self, *group_names):
        for group_name in group_names:
            assert group_name in self.__groups, '{0} is not a valid landmark group'.format(group_name)

        self.group_names = group_names
        self.landmark_numbers = []
        self.diff_numbers = []

        for group_name in self.group_names:
            if group_name in self.__groups:
                self.landmark_numbers += self.__groups[group_name]
            if group_name in self.__diff_groups:
                self.diff_numbers += self.__diff_groups[group_name]

        self.landmark_numbers = list(set(self.landmark_numbers))
        self.diff_numbers = list(set(self.diff_numbers))

    def landmark_number(self, landmark_index):
        if landmark_index < len(self.landmark_numbers):
            return self.landmark_numbers[landmark_index]
        diff_pair = self.diff_numbers[landmark_index-len(self.landmark_numbers)]
        return '{0}/{1}'.format(diff_pair[0], diff_pair[1])

    def number_of_features(self):
        return len(self.landmark_numbers)+len(self.diff_numbers)

    def normalized_symmetric_landmark_differences(self, peak_landmarks, neutral_landmarks):
        assert len(peak_landmarks) == len(neutral_landmarks)

        lm_one = [one for (one,two) in self.diff_numbers]
        lm_two = [two for (one,two) in self.diff_numbers]

        peak_diffs = self.normalized_landmark_differences(peak_landmarks[:, lm_one, :], peak_landmarks[:, lm_two, :])
        neutral_diffs = self.normalized_landmark_differences(neutral_landmarks[:, lm_one, :], neutral_landmarks[:, lm_two, :])

        return (np.abs(peak_diffs-neutral_diffs)).reshape((-1, 2*len(lm_one)))

    def extract_features(self, peak_landmarks, neutral_landmarks):
        """ Return the distance between selected peak and neutral landmarks for x and y direction as features """

        landmark_features = self.normalized_landmark_differences(peak_landmarks[:, self.landmark_numbers, :], neutral_landmarks[:, self.landmark_numbers, :])
        diff_features = self.normalized_symmetric_landmark_differences(peak_landmarks, neutral_landmarks)

        return np.concatenate((landmark_features, diff_features), axis=1)

    def describe(self):
        return 'Normalized X/Y peak/normal differences for landmarks out of {0}'.format(', '.join(self.group_names))
