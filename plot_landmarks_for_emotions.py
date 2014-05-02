#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import os

DATA_DIR = 'data'
IMAGES_DIR = os.path.join(DATA_DIR, 'cohn-kanade-images')
EMOTIONS_DIR = os.path.join(DATA_DIR, 'Emotion')
FACS_DIR = os.path.join(DATA_DIR, 'FACS')
LANDMARKS_DIR = os.path.join(DATA_DIR, 'Landmarks')

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

def main():
    """ Plot all emotions and landmarks """
    colors = {
        0: 'w',
        1: 'g',
        2: 'r',
        3: 'c',
        4: 'm',
        5: 'y',
        6: 'k',
        7: 'b',
        UNKNOWN_EMOTION: '0.1'
    }

    plot_data = { emotion: ([], []) for emotion in EMOTIONS }

    subjects = get_subjects()
    for subject in subjects:
        image_sequences = get_image_sequences(subject)
        for image_sequence in image_sequences:
            emotion = read_emotion(subject, image_sequence)
            X, Y = read_peak_landmarks(subject, image_sequence)

            plot_data[emotion][0].append(X)
            plot_data[emotion][1].append(Y)

    for emotion in EMOTIONS:
        if emotion == UNKNOWN_EMOTION or len(plot_data[emotion][0]) == 0:
            continue

        X = np.concatenate(plot_data[emotion][0])
        Y = np.concatenate(plot_data[emotion][1])
        plt.scatter(X, Y, color=colors[emotion], alpha=0.5, s=20, lw=0, label=EMOTIONS[emotion])

    plt.xlabel('X pixel position of landmark.')
    plt.ylabel('Y pixel position of landmark.')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()
