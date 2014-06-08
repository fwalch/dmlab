#ifndef _PythonWrapper_H_
#define _PythonWrapper_H_

#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_8_API_VERSION
#include <numpy/ndarrayobject.h>
#include <string>
#include <cassert>
#include <iterator>
#include <opencv2/core/core.hpp>

class PythonWrapper
{
  private:
    PyObject* pyModule;
    PyObject* pyFunction;

  public:
    PythonWrapper()
    {
      Py_Initialize();
      PyRun_SimpleString("import sys\nsys.path.append('../..')");
      pyModule = PyImport_ImportModule("predict");
      if (pyModule == nullptr) {
        std::cerr << "Launch the program from its build directory (i.e.`cd bin && ./demo`) or use `make demo`." << std::endl;
        std::exit(-99);
      }
      pyFunction = PyObject_GetAttrString(pyModule, "predict_emotion");
      assert(pyFunction != nullptr && PyCallable_Check(pyFunction));
      []() { import_array(); return 0L; }();
    }

    ~PythonWrapper()
    {
      Py_XDECREF(pyFunction);
      Py_XDECREF(pyModule);
      Py_Finalize();
    }

    PyObject* createNumpyArray(const cv::Mat& mat) const
    {
      assert(mat.type() == CV_64F && mat.isContinuous() && mat.data != nullptr);

      std::array<npy_intp, 2> dims {{ mat.rows, mat.cols }};
      return PyArray_SimpleNewFromData(dims.size(), std::begin(dims), NPY_DOUBLE, reinterpret_cast<void*>(mat.data));
    }

    std::string getEmotion(const cv::Mat& neutralLandmarks, const cv::Mat& peakLandmarks) const
    {
      PyObject* pyNeutralLandmarks = createNumpyArray(neutralLandmarks);
      PyObject* pyPeakLandmarks = createNumpyArray(peakLandmarks);
      PyObject* tuple = PyTuple_Pack(2, pyNeutralLandmarks, pyPeakLandmarks);

      PyObject* pyEmotion = PyObject_CallObject(pyFunction, tuple);
      Py_DECREF(tuple);
      Py_DECREF(pyPeakLandmarks);
      Py_DECREF(pyNeutralLandmarks);
      assert(pyEmotion != nullptr);

      Py_ssize_t emotionLength;
      const char* emotionPtr = PyUnicode_AsUTF8AndSize(pyEmotion, &emotionLength);
      std::string emotion(emotionPtr, static_cast<size_t>(emotionLength));

      Py_DECREF(pyEmotion);
      return emotion;
    }
};

#endif
