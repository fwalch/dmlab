#ifndef _LandmarkExtractor_H_
#define _LandmarkExtractor_H_

#include <opencv2/core/core.hpp>
#include <cassert>

class LandmarkExtractor
{
  private:
    const double MovementThreshold = 1.0; //TODO
    cv::Mat previousLandmarks;
    cv::Mat currentLandmarks;

  public:
    LandmarkExtractor()
    {
      reset();
    }

    bool process(const cv::Mat& image)
    {
      previousLandmarks = currentLandmarks;
      //TODO

      if (previousLandmarks.empty()) {
        return false;
      }

      return cv::norm(previousLandmarks, currentLandmarks) < MovementThreshold;
    }

    const cv::Mat& landmarks() const
    {
      assert(!currentLandmarks.empty());
      return currentLandmarks;
    }

    void reset()
    {
      currentLandmarks = cv::Mat();
      previousLandmarks = cv::Mat();
    }
};

#endif
