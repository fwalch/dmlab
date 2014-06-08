from parser import Parser

class ClassifierWrapper:
    def __init__(self, extractor, classifier, score):
        assert extractor != None
        assert classifier != None

        self.feature_extractor = extractor
        self.classifier = classifier
        self.score = score

    def predict_emotion(self, neutral_landmarks, peak_landmarks):
        features = self.feature_extractor.extract_features(peak_landmarks, neutral_landmarks)
        emotion = self.classifier.predict(features)
        return Parser.emotion_to_str(emotion)

