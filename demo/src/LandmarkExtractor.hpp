#ifndef _LandmarkExtractor_H_
#define _LandmarkExtractor_H_

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cassert>
#include <stasm/stasm_lib.h>

class LandmarkExtractor
{
  private:
    const double MovementThreshold = 2.0; //TODO
    cv::Mat previousLandmarks;
    cv::Mat currentLandmarks;

    bool extractLandmarks(const cv::Mat& image, cv::Mat& landmarks)
    {
      // Convert to grayscale
      cv::Mat grayImage;
      if (image.type() != CV_8UC1) {
        cv::cvtColor(image, grayImage, CV_RGB2GRAY);
      }
      else {
        grayImage = image;
      }

      // Extract landmarks using STASM
      int foundFace;
      landmarks = cv::Mat(stasm_NLANDMARKS, 2, CV_32F);

      if (!stasm_search_single(&foundFace, reinterpret_cast<float*>(landmarks.data), reinterpret_cast<char*>(grayImage.data), grayImage.cols, grayImage.rows, "camera", "../third_party/stasm/data")) {
        std::cerr << "STASM error: " << stasm_lasterr() << std::endl;
        std::cerr << "Launching the program from its build directory (i.e.`cd bin && ./demo`) or using `make demo` might help." << std::endl;
        std::exit(-98);
      }

      return foundFace;
    }

  public:
    LandmarkExtractor()
    {
      reset();
    }

    bool process(cv::Mat& image)
    {
      previousLandmarks = currentLandmarks;

      if (extractLandmarks(image, currentLandmarks)) {
        float* ptr = reinterpret_cast<float*>(currentLandmarks.data);
        for (int i = 0; i < currentLandmarks.rows; i++) {
          cv::circle(image, cv::Point(ptr[i*2], ptr[i*2+1]), 2, CV_RGB(255, 0, 0), 1, CV_AA);
        }
      }

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
