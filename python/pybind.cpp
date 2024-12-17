#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "LU.hpp"
#include "basic.hpp"
#include "ConjugateGradient.hpp"
#include "KrylovSubspace.hpp"
#include "IterSolver.hpp"

namespace py = pybind11;

PYBIND11_MODULE(numerical_solver, m) {
    m.doc() = "Pybind11 interface for numerical solvers and linear algebra tools";

    // LU Decomposition
    m.def("pivot_lu", &LU::PivotLU<double>, "Perform LU decomposition with partial pivoting",
          py::arg("A"), py::arg("P"));

    // Basic Operations
    m.def("power_iter", &basic::powerIter<double>, "Power iteration method",
          py::arg("A"), py::arg("b"), py::arg("max_iter"));
    m.def("rayleigh_quotient", &basic::rayleighQuotient<double>, "Compute Rayleigh quotient",
          py::arg("A"), py::arg("b"));

    // Krylov Subspace
    m.def("arnoldi", &Krylov::Arnoldi<double>, "Arnoldi iteration for Krylov subspace",
          py::arg("A"), py::arg("Q"), py::arg("H"), py::arg("tol"));

    // Conjugate Gradient Solver
    py::class_<ConjugateGrad<double>>(m, "ConjugateGrad")
        .def(py::init<>())
        .def(py::init<MatrixObj<double>, MatrixObj<double>, VectorObj<double>, int, double>())
        .def("call_update", &ConjugateGrad<double>::callUpdate, "Run the conjugate gradient solver");

    // Iterative Solvers
    py::class_<GradientDescent<double>>(m, "GradientDescent")
        .def(py::init<>())
        .def(py::init<MatrixObj<double>, MatrixObj<double>, VectorObj<double>, int>())
        .def("solve", &GradientDescent<double>::solve, "Solve system using gradient descent");

    py::class_<StaticIterMethod<double>>(m, "StaticIterMethod")
        .def(py::init<>())
        .def(py::init<MatrixObj<double>, MatrixObj<double>, VectorObj<double>, int>())
        .def("solve", &StaticIterMethod<double>::solve, "Solve system using static iterative method");
}
