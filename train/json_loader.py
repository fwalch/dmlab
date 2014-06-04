from parser import Parser
import os
import json
import numpy as np

class JsonLoader:
    def __init__(self, base_dir='.'):
        self.parser = Parser()
        self.json_path = os.path.join(base_dir, self.parser.data_dir(), 'training.json')

    def dump_json(self):
        print('Exporting data to JSON')
        neutral_lm = []
        peak_lm = []
        emotions = []

        def callback(subject, image_sequence, facs, emotion, neutral_landmarks, peak_landmarks):
            neutral_lm.append(neutral_landmarks.tolist())
            peak_lm.append(peak_landmarks.tolist())
            emotions.append(emotion)

        self.parser.process_data(callback)

        with open(self.json_path, 'w') as outfile:
            json.dump({'neutral_landmarks': neutral_lm, 'peak_landmarks': peak_lm, 'emotions': emotions}, outfile, indent=2)

    def load(self, only_labeled=True):
        if not os.path.isfile(self.json_path):
            self.dump_json()

        with open(self.json_path) as infile:
            data = json.load(infile)
            neutral_landmarks = np.array(data['neutral_landmarks'], dtype=float)
            peak_landmarks = np.array(data['peak_landmarks'], dtype=float)
            emotions = np.array(data['emotions'], dtype=int)
            del data

        if only_labeled:
            labeled = np.logical_not(self.parser.is_unknown_emotion(emotions))
            neutral_landmarks = neutral_landmarks[labeled]
            peak_landmarks = peak_landmarks[labeled]
            emotions = emotions[labeled]

        assert len(neutral_landmarks) == len(peak_landmarks)
        assert len(neutral_landmarks) == len(emotions)

        assert neutral_landmarks.shape[1] == 68

        return neutral_landmarks, peak_landmarks, emotions

    @classmethod
    def load_default(cls):
        return cls().load()
