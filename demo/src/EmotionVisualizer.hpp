#ifndef _EmotionVisualizer_H_
#define _EmotionVisualizer_H_

#include <opencv2/highgui/highgui.hpp>
#include <map>

class EmotionVisualizer
{
  private:
    std::map<std::string, cv::Mat> emotionImages;

  public:
    EmotionVisualizer()
    {
      emotionImages["anger"] = cv::imread("../emoticons/anger.png", -1);
      emotionImages["contempt"] = cv::imread("../emoticons/contempt.png", -1);
      emotionImages["disgust"] = cv::imread("../emoticons/disgust.png", -1);
      emotionImages["fear"] = cv::imread("../emoticons/fear.png", -1);
      emotionImages["happiness"] = cv::imread("../emoticons/happiness.png", -1);
      emotionImages["sadness"] = cv::imread("../emoticons/sadness.png", -1);
      emotionImages["surprise"] = cv::imread("../emoticons/surprise.png", -1);
    }

    void drawEmotion(cv::Mat& image, std::string emotion, const cv::Mat& landmarks)
    {
      int bottomOfFaceX = landmarks.at<float>(36, 0);
      int bottomOfFaceY = landmarks.at<float>(36, 1);
      cv::putText(image, emotion, cv::Point(bottomOfFaceX-70*image.cols/640, bottomOfFaceY+50*image.rows/480), cv::FONT_HERSHEY_SIMPLEX, 2*image.cols/640, CV_RGB(0, 255, 0), 2*image.cols/640);
    }

    void drawEmotionAndSmilie(cv::Mat& image, std::string emotion, const cv::Mat& landmarks)
    {
      drawEmotion(image, emotion, landmarks);

      int upperX = landmarks.at<float>(35, 0);
      //int upperY = landmarks.at<float>(1, 1);
      int upperY = landmarks.at<float>(38, 1);
      int lowerX = landmarks.at<float>(37, 0);
      int lowerY = landmarks.at<float>(36, 1);

      if (upperX >= lowerX || upperY >= lowerY || lowerX >= image.cols || lowerY >= image.rows || upperX <= 0 || upperY <= 0)
        return;

      //TODO: this is not optimized
      cv::Rect faceRegion(upperX, upperY, lowerX-upperX, lowerY-upperY);
      cv::Mat rgba = emotionImages[emotion];
      cv::Mat bgr( rgba.rows, rgba.cols, CV_8UC3 );
      cv::Mat alpha( rgba.rows, rgba.cols, CV_8UC1 );

      cv::Mat out[] = { bgr, alpha };
      int from_to[] = { 0,0, 1,1, 2,2, 3,3 };
      cv::mixChannels( &rgba, 1, out, 2, from_to, 4 );

      cv::Mat smallMask;
      cv::compare(alpha, 255, smallMask, cv::CMP_EQ);

      cv::Mat face, mask;
      cv::resize(bgr, face, faceRegion.size(), 1, 1, cv::INTER_LINEAR);
      cv::resize(smallMask, mask, faceRegion.size(), 1, 1, cv::INTER_LINEAR);

      cv::Mat roi(image(faceRegion));
      face.copyTo(roi, mask);
    }
};

#endif
