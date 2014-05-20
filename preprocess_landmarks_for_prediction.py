#!/usr/bin/env python3

import json
import os
from parser import Parser

if __name__ == '__main__':
    p = Parser()

    neutral_lm = []
    peak_lm = []
    emotions = []

    def callback(subject, image_sequence, facs, emotion, neutral_landmarks, peak_landmarks):
        neutral_lm.append(neutral_landmarks.tolist())
        peak_lm.append(peak_landmarks.tolist())
        emotions.append(emotion)

    p.process_data(callback)

    with open(os.path.join(p.data_dir(), 'preprocessed.json'), 'w') as outfile:
        json.dump({'neutral_landmarks': neutral_lm, 'peak_landmarks': peak_lm, 'emotions': emotions}, outfile, indent=2)
