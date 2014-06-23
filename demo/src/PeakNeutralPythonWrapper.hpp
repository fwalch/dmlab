#ifndef _PeakNeutralPythonWrapper_H_
#define _PeakNeutralPythonWrapper_H_

#include "PythonWrapper.hpp"

class PeakNeutralPythonWrapper : public PythonWrapper
{
  public:
    PeakNeutralPythonWrapper() : PythonWrapper("predict-neutral-peak")
    {}

    std::string getEmotion(const cv::Mat& neutralLandmarks, const cv::Mat& peakLandmarks) const
    {
      cv::Mat neutralLandmarksClone = neutralLandmarks.clone();
      PyObject* pyNeutralLandmarks = createNumpyArray(neutralLandmarksClone);
      cv::Mat peakLandmarksClone = peakLandmarks.clone();
      PyObject* pyPeakLandmarks = createNumpyArray(peakLandmarksClone);
      PyObject* tuple = PyTuple_Pack(2, pyNeutralLandmarks, pyPeakLandmarks);

      PyObject* pyEmotion = PyObject_CallObject(pyFunction, tuple);
      Py_DECREF(tuple);
      Py_DECREF(pyNeutralLandmarks);
      Py_DECREF(pyPeakLandmarks);
      assert(pyEmotion != nullptr);

      const char* emotionPtr = PyUnicode_AsUTF8(pyEmotion);
      std::string emotion(emotionPtr);
      std::cout << emotion << std::endl;

      Py_DECREF(pyEmotion);
      return emotion;
    }

};

#endif
