#ifndef _EmotionPredictor_H_
#define _EmotionPredictor_H_

#include <opencv2/core/core.hpp>
#include "PythonWrapper.hpp"

class EmotionPredictor
{
  private:
    PythonWrapper wrapper;

  public:
    void process(cv::Mat& image, const cv::Mat& neutralLandmarks, const cv::Mat& peakLandmarks) const
    {
      std::string emotion = wrapper.getEmotion(neutralLandmarks, peakLandmarks);
      //TODO: replace image with smiley
      cv::putText(image, emotion, cv::Point(image.rows/2, image.cols/2), cv::FONT_HERSHEY_DUPLEX, 1, CV_RGB(255, 0, 0));
    }
};

#endif
