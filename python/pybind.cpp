#include <pybind11/pybind11.h>
#include "MatrixObj.hpp"
#include "VectorObj.hpp"
#include "IterSolver.hpp"
#include "SolverBase.hpp"

namespace py = pybind11;

PYBIND11_MODULE(matrixobj, m) {
    py::class_<MatrixObj<double>>(m, "MatrixObj")
        .def(py::init<>())
        .def(py::init<int, int>())
        .def(py::init<const double*, int, int>())
        .def("get_row", &MatrixObj<double>::get_row)
        .def("get_col", &MatrixObj<double>::get_col)
        // 绑定其他成员函数和属性
        ;

    py::class_<VectorObj<double>, MatrixObj<double>>(m, "VectorObj")
        .def(py::init<>())
        .def(py::init<int>())
        .def(py::init<const double*, int>())
        .def("L2norm", &VectorObj<double>::L2norm)
        .def("normalized", &VectorObj<double>::normalized)
        // 绑定其他成员函数和属性
        ;

    py::class_<IterSolverBase<double>>(m, "IterSolverBase")
        .def(py::init<>())
        .def(py::init<MatrixObj<double>, MatrixObj<double>, VectorObj<double>, int>())
        // 绑定其他成员函数和属性
        ;

    py::class_<GradientDesent<double>, IterSolverBase<double>>(m, "GradientDesent")
        .def(py::init<>())
        .def(py::init<MatrixObj<double>, MatrixObj<double>, VectorObj<double>, int>())
        // 绑定其他成员函数和属性
        ;

    py::class_<Jacobi<double>, IterSolverBase<double>>(m, "Jacobi")
        .def(py::init<>())
        .def(py::init<MatrixObj<double>, MatrixObj<double>, VectorObj<double>, int>())
        // 绑定其他成员函数和属性
        ;
}