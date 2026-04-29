#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include "../Ridge/Ridge.cpp"

namespace py = pybind11;

// Pythonに「SimpleML」というモジュールとして公開する
PYBIND11_MODULE(SimpleML, m) {
    // StandardScalerクラスをPythonに登録
    py::class_<StandardScaler>(m, "StandardScaler")
        .def(py::init<>()) // コンストラクタ
        .def("fit", &StandardScaler::fit)
        .def("transform", &StandardScaler::transform)
        .def("fit_transform", &StandardScaler::fit_transform);

    // RidgeRegressorクラスをPythonに登録
    py::class_<RidgeRegressor>(m, "RidgeRegressor")
        .def(py::init<double>(), py::arg("lambda") = 1.0)
        .def("fit", &RidgeRegressor::fit)
        .def("predict", &RidgeRegressor::predict)
        .def("get_beta", &RidgeRegressor::get_beta);
}