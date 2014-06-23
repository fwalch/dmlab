#ifndef _EmotionPredictor_H_
#define _EmotionPredictor_H_

#include <opencv2/core/core.hpp>
#include "PeakNeutralPythonWrapper.hpp"
#include "EmotionVisualizer.hpp"

class EmotionPredictor
{
  private:
    PeakNeutralPythonWrapper wrapper;
    EmotionVisualizer visualizer;
    std::string emotion;

  public:
    void process(const cv::Mat& image, const cv::Mat& neutralLandmarks, const cv::Mat& peakLandmarks)
    {
      emotion = wrapper.getEmotion(neutralLandmarks, peakLandmarks);
    }

    void drawEmotion(cv::Mat& image, const cv::Mat& landmarks)
    {
      visualizer.drawEmotionAndSmilie(image, emotion, landmarks);
    }

    void reset()
    {
      emotion = "";
    }
};

#endif
