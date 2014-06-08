#!/usr/bin/python3

import os
import pickle

def predict_emotion(neutral_landmarks, peak_landmarks):
    assert os.path.exists('classifier.pkl'), 'Classifier not saved; run train.py with the --persist option'

    #TODO: don't load file for each prediction
    with open('classifier.pkl', 'rb') as infile:
        classifier = pickle.load(infile)

    return classifier.predict_emotion(neutral_landmarks, peak_landmarks)

if __name__ == '__main__':
    # TODO: args: image sequence
    # Load first/last image, predict, compare with real emotion
    raise Exception('Not a standalone program (yet)')
