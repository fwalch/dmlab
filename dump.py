#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import json
import os

DATA_DIR = 'data'
IMAGES_DIR = os.path.join(DATA_DIR, 'cohn-kanade-images')
EMOTIONS_DIR = os.path.join(DATA_DIR, 'Emotion')
FACS_DIR = os.path.join(DATA_DIR, 'FACS')
LANDMARKS_DIR = os.path.join(DATA_DIR, 'Landmarks')

def get_subjects():
    return sorted([subject for subject in os.listdir(IMAGES_DIR) if subject.startswith('S')])

def get_image_sequences(subject):
    return sorted([seq for seq in os.listdir(os.path.join(IMAGES_DIR, subject)) if seq.startswith('0')])

def __load_single_file(directory):
    if not os.path.isdir(directory):
        return None

    files = os.listdir(directory)
    file_count = len(files)

    assert file_count == 0 or file_count == 1

    if file_count == 0:
        return None

    return os.path.join(directory, files[0])

def read_emotion(subject, image_sequence):
    emotion_dir = os.path.join(EMOTIONS_DIR, subject, image_sequence)
    emotion_file_path = __load_single_file(emotion_dir)

    if not emotion_file_path:
        return None

    return int(np.loadtxt(emotion_file_path))

def read_facs(subject, image_sequence):
    facs_dir = os.path.join(FACS_DIR, subject, image_sequence)
    facs_file_path = __load_single_file(facs_dir)

    if not facs_file_path:
        return None

    return np.loadtxt(facs_file_path).reshape(-1, 2)

def read_peak_landmarks(subject, image_sequence):
    landmarks_dir = os.path.join(LANDMARKS_DIR, subject, image_sequence)
    landmark_file_path = os.path.join(landmarks_dir, sorted(os.listdir(landmarks_dir))[-1])
    return np.loadtxt(landmark_file_path)

def load_image(subject, image_sequence, image):
    return img.imread(os.path.join(IMAGES_DIR, subject, image_sequence, image))

def read_images(subject, image_sequence):
    images = []
    for image in os.listdir(os.path.join(IMAGES_DIR, subject, image_sequence)):
        if not image.endswith('.png'):
            continue
        #images.append({'name': image', 'data': load_image(subject, image_sequence, image)})
        images.append({'name': image})
    return images

def main():
    """ Load all data, dump as JSON """

    subjects = get_subjects()

    flat_data = {'subjects': [], 'facs': [], 'image_sequences': [], 'emotions': [], 'landmarks': []}

    for subject in subjects:
        flat_data['subjects'].append({'name': subject})

        image_sequences = get_image_sequences(subject)
        for image_sequence in image_sequences:
            flat_data['image_sequences'].append({'subject': subject, 'name': image_sequence})

            facs = read_facs(subject, image_sequence)
            for au, intensity in facs:
                flat_data['facs'].append({'au': int(au), 'intensity': intensity, 'subject': subject, 'image_sequence': image_sequence})

            emotion = read_emotion(subject, image_sequence)
            flat_data['emotions'].append({'emotion': emotion, 'subject': subject, 'image_sequence': image_sequence})

            landmarks = read_peak_landmarks(subject, image_sequence)
            for x, y in landmarks:
                flat_data['landmarks'].append({'subject': subject, 'image_sequence': image_sequence, 'x': x, 'y': y})


    with open(os.path.join(DATA_DIR, 'data.json'), 'w') as outfile:
        json.dump(flat_data, outfile, indent=2)

    for key in flat_data:
        with open(os.path.join(DATA_DIR, '{0}.json'.format(key)), 'w') as outfile:
            json.dump(flat_data[key], outfile, indent=2)

if __name__ == '__main__':
    main()
