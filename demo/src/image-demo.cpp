#include <cstdlib>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "LandmarkExtractor.hpp"
#include "PeakPythonWrapper.hpp"
#include "EmotionVisualizer.hpp"

int main(int argc, char** argv)
{
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " path_to_image" << std::endl;
    return -1;
  }

  cv::Mat image;
  image = cv::imread(argv[1], CV_LOAD_IMAGE_ANYCOLOR);

  if (!image.data) {
    std::cerr << "Image not found." << std::endl;
    return -2;
  }

  std::cout << "Extracting landmarks." << std::endl;
  LandmarkExtractor landmarkExtractor;
  cv::Mat landmarks;
  if (!landmarkExtractor.extractLandmarks(image, landmarks)) {
    std::cerr << "No landmarks found." << std::endl;
    return -3;
  }
  landmarkExtractor.drawLandmarks(image, landmarks);

  std::cout << "Predicting emotion." << std::endl;
  PeakPythonWrapper peakWrapper;
  std::string emotion = peakWrapper.getEmotion(landmarks);
  EmotionVisualizer visualizer;
  std::cout << "Visualizing emotion." << std::endl;
  //visualizer.drawEmotion(image, emotion, landmarks);

  cv::namedWindow("Emotions", CV_WINDOW_NORMAL | CV_WINDOW_KEEPRATIO | CV_GUI_EXPANDED);
  cv::imshow("Emotions", image);
  cv::imwrite("out.png", image);

  cv::waitKey(0);
  cv::destroyAllWindows();
}
