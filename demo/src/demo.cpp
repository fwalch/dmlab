#include <cstdlib>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "Demo.hpp"

int main(int argc, char** argv)
{
  if (argc > 2) {
    std::cerr << "Usage: " << argv[0] << " [camera_id]" << std::endl;
    return -1;
  }

  int cameraId = argc == 2 ? std::atoi(argv[2]) : 0;

  cv::VideoCapture camera(cameraId);
  if (!camera.isOpened()) {
    std::cerr << "Camera ID " << cameraId << " invalid." << std::endl;
    return -2;
  }

  // Initialize OpenCV output windows
  int width = static_cast<int>(camera.get(CV_CAP_PROP_FRAME_WIDTH));
  int height = static_cast<int>(camera.get(CV_CAP_PROP_FRAME_HEIGHT));
  const cv::Mat defaultImage = cv::Mat::zeros(height, width, CV_8U);

  Demo demo;
  for (const std::string& outputWindow : demo.OutputWindows) {
    cv::namedWindow(outputWindow, CV_WINDOW_NORMAL | CV_WINDOW_KEEPRATIO | CV_GUI_EXPANDED);
    cv::imshow(outputWindow, defaultImage);
  }

  std::cout << "Press 'q' to quit, 's' to start emotion prediction, 'r' to reset prediction process." << std::endl;

  while (true) {
    cv::Mat image;
    camera >> image;

    std::string window;
    if (demo.process(image, window)) {
      cv::imshow(window, image);
    }

    int pressedKey = cv::waitKey(1) & 0xFF;
    switch (pressedKey) {
      case 'q':
        std::cout << "Exiting application." << std::endl;
        cv::destroyAllWindows();
        return 0;

      case 's':
        demo.start();
        break;

      case 'r':
        demo.reset();

        for (const std::string& outputWindow : demo.OutputWindows) {
          cv::imshow(outputWindow, defaultImage);
        }
        break;
    }
  }
}
