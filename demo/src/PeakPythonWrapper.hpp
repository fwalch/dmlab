#ifndef _PeakPythonWrapper_H_
#define _PeakPythonWrapper_H_

#include "PythonWrapper.hpp"

class PeakPythonWrapper : public PythonWrapper
{
  public:
    PeakPythonWrapper() : PythonWrapper("predict-peak")
    {}

    std::string getEmotion(const cv::Mat& peakLandmarks) const
    {
      cv::Mat landmarks = peakLandmarks.clone();
      PyObject* pyPeakLandmarks = createNumpyArray(landmarks);

      PyObject* tuple = PyTuple_Pack(1, pyPeakLandmarks);
      PyObject* pyEmotion = PyObject_CallObject(pyFunction, tuple);
      Py_DECREF(tuple);
      Py_DECREF(pyPeakLandmarks);
      assert(pyEmotion != nullptr);

      const char* emotionPtr = PyUnicode_AsUTF8(pyEmotion);
      std::string emotion(emotionPtr);

      Py_DECREF(pyEmotion);
      return emotion;
    }
};

#endif
