#ifndef _LandmarkExtractor_H_
#define _LandmarkExtractor_H_

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cassert>
#include <stasm/stasm_lib.h>

class LandmarkExtractor
{
  private:
    const double MovementThreshold = 22.0;
    const std::array<int, 39> landmarkIndices {{
      18,21,22,25,58,57,56,55,54,34,30,40,44,59,60,61,62,63,64,65,72,73,74,75,76,68,67,66,71,70,69,17,16,23,24,0,6,12,14
    }};
    cv::Mat previousLandmarks;
    cv::Mat currentLandmarks;

  public:
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
      cv::Mat fullLandmarks = cv::Mat(stasm_NLANDMARKS, 2, CV_32F);

      if (!stasm_search_single(&foundFace, reinterpret_cast<float*>(fullLandmarks.data), reinterpret_cast<char*>(grayImage.data), grayImage.cols, grayImage.rows, "camera", "../third_party/stasm/data")) {
        std::cerr << "STASM error: " << stasm_lasterr() << std::endl;
        std::cerr << "Launching the program from its build directory (i.e.`cd bin && ./demo`) or using `make demo` might help." << std::endl;
        std::exit(-98);
      }

      landmarks = cv::Mat(landmarkIndices.size(), 2, CV_32F);
      for (int i = 0; i < static_cast<int>(landmarkIndices.size()); i++) {
        fullLandmarks.row(landmarkIndices[i]).copyTo(landmarks.row(i));
      }

      return foundFace;
    }

    LandmarkExtractor()
    {
      reset();
    }

    void drawLandmarks(cv::Mat& image, const cv::Mat& landmarks)
    {
      assert(!landmarks.empty());
      float* ptr = reinterpret_cast<float*>(landmarks.data);
      for (int i = 0; i < landmarks.rows; i++) {
        cv::circle(image, cv::Point(ptr[i*2], ptr[i*2+1]), 2*image.cols/640, CV_RGB(255, 0, 0), -1, CV_AA);
      }
  }

    bool process(cv::Mat& image)
    {
      previousLandmarks = currentLandmarks;

      if (extractLandmarks(image, currentLandmarks)) {
        drawLandmarks(image, currentLandmarks);
      }

      if (previousLandmarks.empty()) {
        return false;
      }

      float norm = cv::norm(previousLandmarks, currentLandmarks);
      return norm < MovementThreshold;
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
