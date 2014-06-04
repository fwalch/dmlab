import numpy as np
from sklearn import preprocessing

class NormalizedLandmarkDistances:
    """ Base class for features based on normalized peak/neutral landmark coordinate differences """

    def feature_names(self):
        return ['normalized distance peak/neutral (x)', 'normalized distance peak/neutral (y)']

    def __normalize_landmarks(self, landmarks):
        """ Normalize for each face individually """
        for image_sequence_number in range(len(landmarks)):
            landmarks[image_sequence_number, :, :] = preprocessing.scale(landmarks[image_sequence_number, :, :])

        return landmarks

    def normalized_landmark_differences(self, peak_landmarks, neutral_landmarks):
        assert len(peak_landmarks) == len(neutral_landmarks)
        number_of_landmarks = peak_landmarks.shape[1]
        number_of_features = 2*number_of_landmarks # 2 features (x and y) for each landmark

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

    def extract_features(self, peak_landmarks, neutral_landmarks):
        """ Return the distance between selected peak and neutral landmarks for x and y direction as features """
        landmark_numbers = []
        for group_name in self.group_names:
            landmark_numbers += self.__groups[group_name]

        return self.normalized_landmark_differences(peak_landmarks[:, landmark_numbers, :], neutral_landmarks[:, landmark_numbers, :])

    def describe(self):
        return 'Normalized X/Y peak/normal differences for landmarks out of {0}'.format(', '.join(self.group_names))
