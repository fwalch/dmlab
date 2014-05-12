#!/usr/bin/env python3

# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt
import os

DATA_DIR = 'data'
IMAGES_DIR = os.path.join(DATA_DIR, 'cohn-kanade-images')
EMOTIONS_DIR = os.path.join(DATA_DIR, 'Emotion_labels/Emotion')
FACS_DIR = os.path.join(DATA_DIR, 'FACS')
LANDMARKS_DIR = os.path.join(DATA_DIR, 'Landmarks/Landmarks')

UNKNOWN_EMOTION = -1
EMOTIONS = {
  0: 'neutral',
  1: 'anger',
  2: 'contempt',
  3: 'disgust',
  4: 'fear',
  5: 'happiness',
  6: 'sadness',
  7: 'surprise',

  UNKNOWN_EMOTION: 'unknown'
}

def get_subjects():
    return sorted([subject for subject in os.listdir(IMAGES_DIR) if subject.startswith('S')])

def get_image_sequences(subject):
    return sorted([seq for seq in os.listdir(os.path.join(IMAGES_DIR, subject)) if seq.startswith('0')])

def read_emotion(subject, image_sequence):
    emotion_dir = os.path.join(EMOTIONS_DIR, subject, image_sequence)

    if not os.path.isdir(emotion_dir):
        return UNKNOWN_EMOTION

    emotion_files = os.listdir(emotion_dir)
    emotion_files_count = len(emotion_files)

    assert emotion_files_count == 0 or emotion_files_count == 1

    if emotion_files_count == 0:
        return UNKNOWN_EMOTION

    emotion_file_path = os.path.join(emotion_dir, emotion_files[0])
    return int(np.loadtxt(emotion_file_path))

def read_peak_landmarks(subject, image_sequence):
    landmarks_dir = os.path.join(LANDMARKS_DIR, subject, image_sequence)
    landmark_file_path = os.path.join(landmarks_dir, sorted(os.listdir(landmarks_dir))[-1])


    return np.loadtxt(landmark_file_path, unpack=True)

def read_neutral_landmarks(subject, image_sequence):
    landmarks_dir = os.path.join(LANDMARKS_DIR, subject, image_sequence)
    landmark_file_path = os.path.join(landmarks_dir, sorted(os.listdir(landmarks_dir))[0])

    if landmark_file_path[-1] !='t':
        landmark_file_path = os.path.join(landmarks_dir, sorted(os.listdir(landmarks_dir))[1])


    return np.loadtxt(landmark_file_path, unpack=True)

def compute_2D_euclidean_dist(X1, Y1, X2, Y2):
    distTot = []
    i = 0
    for l in X1:
                x1 = X1[i]
                y1 = Y1[i]

                x2 = X2[i]
                y2 = Y2[i]

                a = np.array((x1,y1))
                b = np.array((x2,y2))

                dist = np.linalg.norm(a-b) #compute euclidean distance 68x1 matrix
                distTot.append(dist)
                i= i+1

    return distTot

def main():

    datasetX = []
    datasetY = []


    subjects = get_subjects()
    for subject in subjects:
        image_sequences = get_image_sequences(subject)

        for image_sequence in image_sequences:
            emotion = read_emotion(subject, image_sequence)

            if emotion != -1:
                X1, Y1 = read_neutral_landmarks(subject, image_sequence)
                X2, Y2 = read_peak_landmarks(subject, image_sequence)

                distTot = compute_2D_euclidean_dist(X1, Y1, X2, Y2)


                #print distTot
                datasetX.append(distTot)
                datasetY.append(emotion)




    np.array(datasetX).dump(open('datasetX.npy', 'wb'))
    np.array(datasetY).dump(open('datasetY.npy', 'wb'))




if __name__ == '__main__':
    main()
