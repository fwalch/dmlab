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
  protected:
    PyObject* pyModule;
    PyObject* pyFunction;

    PyObject* createNumpyArray(const cv::Mat& mat) const
    {
      assert(mat.type() == CV_64F && mat.isContinuous() && mat.data != nullptr);

      std::array<npy_intp, 2> dims {{ mat.rows-1, mat.cols }};
      return PyArray_SimpleNewFromData(dims.size(), std::begin(dims), NPY_FLOAT, reinterpret_cast<void*>(mat.data));
    }

  public:
    PythonWrapper(const char* moduleName)
    {
      Py_Initialize();
      PyRun_SimpleString("import sys;sys.path.append('../..')");
      pyModule = PyImport_ImportModule(moduleName);
      if (pyModule == nullptr) {
        std::cerr << "Python module import error." << std::endl;
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
};

#endif
