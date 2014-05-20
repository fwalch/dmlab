import os
import numpy as np
import matplotlib.image as img

class Parser:
    """ gets dataset from files into memory """

    __DATA_DIR = 'data'
    __IMAGES_DIR = os.path.join(__DATA_DIR, 'cohn-kanade-images')
    __EMOTIONS_DIR = os.path.join(__DATA_DIR, 'Emotion')
    __FACS_DIR = os.path.join(__DATA_DIR, 'FACS')
    __LANDMARKS_DIR = os.path.join(__DATA_DIR, 'Landmarks')
    __EMOTIONS = {
      0: 'neutral',
      1: 'anger',
      2: 'contempt',
      3: 'disgust',
      4: 'fear',
      5: 'happiness',
      6: 'sadness',
      7: 'surprise',
      None: 'unknown'
    }

    @classmethod
    def data_dir(cls):
        return cls.__DATA_DIR

    @classmethod
    def emotion_to_str(cls, emotion):
        return cls.__EMOTIONS[emotion]

    def get_subjects(self):
        return sorted([subject for subject in os.listdir(self.__IMAGES_DIR) if subject.startswith('S')])

    def get_image_sequences(self, subject):
        return sorted([seq for seq in os.listdir(os.path.join(self.__IMAGES_DIR, subject)) if seq.startswith('0')])

    def __get_single_file(self, directory):
        if not os.path.isdir(directory):
            return None

        files = os.listdir(directory)
        file_count = len(files)

        assert file_count == 0 or file_count == 1

        if file_count == 0:
            return None

        return os.path.join(directory, files[0])

    def get_emotion(self, subject, image_sequence):
        emotion_dir = os.path.join(self.__EMOTIONS_DIR, subject, image_sequence)
        emotion_file_path = self.__get_single_file(emotion_dir)

        if not emotion_file_path:
            return None

        return int(np.loadtxt(emotion_file_path))

    def get_facs(self, subject, image_sequence):
        facs_dir = os.path.join(self.__FACS_DIR, subject, image_sequence)
        facs_file_path = self.__get_single_file(facs_dir)

        if not facs_file_path:
            return None

        return np.loadtxt(facs_file_path).reshape(-1, 2)

    def __get_landmarks(self, subject, image_sequence, landmark_idx):
        landmarks_dir = os.path.join(self.__LANDMARKS_DIR, subject, image_sequence)
        landmark_file_path = os.path.join(landmarks_dir, sorted([f for f in os.listdir(landmarks_dir) if f.endswith('.txt')])[landmark_idx])
        return np.loadtxt(landmark_file_path)

    def get_peak_landmarks(self, subject, image_sequence):
        return self.__get_landmarks(subject, image_sequence, -1)

    def get_neutral_landmarks(self, subject, image_sequence):
        return self.__get_landmarks(subject, image_sequence, 0)

    def __get_image_data(self, subject, image_sequence, image):
        return img.imget(os.path.join(self.__IMAGES_DIR, subject, image_sequence, image))

    def get_images(self, subject, image_sequence, append_image_data=False):
        images = []
        for image in os.listdir(os.path.join(IMAGES_DIR, subject, image_sequence)):
            if not image.endswith('.png'):
                continue

            if append_image_data:
                images.append({'name': image, 'data': self.__get_image_data(subject, image_sequence, image)})

            images.append({'name': image})
        return images

    def process_data(self, callback):
        subjects = self.get_subjects()
        for subject in subjects:
            image_sequences = self.get_image_sequences(subject)
            for image_sequence in image_sequences:
                facs = self.get_facs(subject, image_sequence)
                emotion = self.get_emotion(subject, image_sequence)
                neutral_landmarks = self.get_neutral_landmarks(subject, image_sequence)
                peak_landmarks = self.get_peak_landmarks(subject, image_sequence)
                callback(subject, image_sequence, facs, emotion, neutral_landmarks, peak_landmarks)
