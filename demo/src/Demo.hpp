#ifndef _Demo_H_
#define _Demo_H_

#include <array>
#include <exception>
#include <opencv2/core/core.hpp>
#include "LandmarkExtractor.hpp"
#include "EmotionPredictor.hpp"

class Demo
{
  private:
    LandmarkExtractor neutralLandmarkExtractor;
    LandmarkExtractor peakLandmarkExtractor;
    EmotionPredictor emotionPredictor;

    enum class State : size_t {
      WaitingForNeutral = 0,
      ExtractNeutralLandmarks = 1,
      WaitingForPeak = 2,
      ExtractPeakLandmarks = 3,
      PredictEmotion = 4,
      PredictionFinished = 5
    };
    State currentState;

  public:
    const std::array<std::string, 3> OutputWindows {{
      "Neutral face",
      "Emotional face",
      "Predicted emotion"
    }};

    Demo()
    {
      reset();
    }

    void changeState(State nextState, std::string& window)
    {
      window = OutputWindows[static_cast<size_t>(currentState)/2];
      currentState = nextState;
    }

    void keepState(std::string& window)
    {
      window = OutputWindows[static_cast<size_t>(currentState)/2];
    }

    void process(cv::Mat& image, std::string& window)
    {
      switch (currentState) {
        case State::WaitingForNeutral:
        case State::WaitingForPeak:
          // Do nothing
          break;

        case State::ExtractNeutralLandmarks:
          if (neutralLandmarkExtractor.process(image)) {
            std::cout << "Finished; waiting for command to extract peak expression landmarks." << std::endl;
            return changeState(State::WaitingForPeak, window);
          }
          break;

        case State::ExtractPeakLandmarks:
          if (peakLandmarkExtractor.process(image)) {
            std::cout << "Switching to emotion prediction." << std::endl;
            return changeState(State::PredictEmotion, window);
          }
          break;

        case State::PredictEmotion:
          emotionPredictor.process(
            image,
            neutralLandmarkExtractor.landmarks(),
            peakLandmarkExtractor.landmarks()
          );
          std::cout << "Prediction finished." << std::endl;
          return changeState(State::PredictionFinished, window);

        case State::PredictionFinished:
          cv::Mat landmarks;
          neutralLandmarkExtractor.extractLandmarks(image, landmarks);
          emotionPredictor.drawEmotion(image, landmarks);
          break;
      }

      return keepState(window);
    }

    void start()
    {
      if (currentState != State::WaitingForNeutral && currentState != State::WaitingForPeak) {
        return;
      }

      if (currentState == State::WaitingForNeutral) {
        std::cout << "Switching to neutral expression landmark extraction." << std::endl;
        currentState = State::ExtractNeutralLandmarks;
      }
      if (currentState == State::WaitingForPeak) {
        std::cout << "Switching to peak expression landmark extraction." << std::endl;
        currentState = State::ExtractPeakLandmarks;
      }
    }

    void reset()
    {
      currentState = State::WaitingForNeutral;
      neutralLandmarkExtractor.reset();
      peakLandmarkExtractor.reset();
      emotionPredictor.reset();
    }
};

#endif
