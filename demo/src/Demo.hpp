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
      WaitingForCommands = 0,
      ExtractNeutralLandmarks = 1,
      ExtractPeakLandmarks = 2,
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

    bool changeState(State nextState, std::string& window)
    {
      window = OutputWindows[static_cast<size_t>(currentState)/2];
      currentState = nextState;
      return true;
    }

    bool keepState(std::string& window)
    {
      window = OutputWindows[static_cast<size_t>(currentState)/2];
      return currentState != State::PredictionFinished;
    }

    bool process(cv::Mat& image, std::string& window)
    {
      switch (currentState) {
        case State::WaitingForCommands:
        case State::PredictionFinished:
          // Do nothing
          break;

        case State::ExtractNeutralLandmarks:
          if (neutralLandmarkExtractor.process(image)) {
            std::cout << "Switching to peak expression landmark extraction." << std::endl;
            return changeState(State::ExtractPeakLandmarks, window);
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
      }

      return keepState(window);
    }

    void start()
    {
      if (currentState != State::WaitingForCommands) {
        return;
      }

      std::cout << "Switching to neutral expression landmark extraction." << std::endl;
      currentState = State::ExtractNeutralLandmarks;
    }

    void reset()
    {
      currentState = State::WaitingForCommands;
      neutralLandmarkExtractor.reset();
      peakLandmarkExtractor.reset();
    }
};

#endif
