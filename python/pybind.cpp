#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "LU.hpp"
#include "basic.hpp"
#include "ConjugateGradient.hpp"
#include "KrylovSubspace.hpp"
#include "IterSolver.hpp"
#include "MultiGrid.hpp"
#include "GMRES.hpp"
#include "RungeKutta.hpp"
#include "GaussianQuad.hpp"

using namespace Quadrature;
namespace py = pybind11;

// 自定义异常
class NumericalSolverError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

PYBIND11_MODULE(numerical_solver, m) {
    m.doc() = "Pybind11 interface for numerical solvers and linear algebra tools";

    // 注册自定义异常
    py::register_exception<NumericalSolverError>(m, "NumericalSolverError");

    // Basic Operations
    m.def("power_iter", &basic::powerIter<double, DenseObj<double>>, 
          "Power iteration method for computing the dominant eigenvalue of a matrix.",
          py::arg("A"), py::arg("b"), py::arg("max_iter"));

    m.def("rayleigh_quotient", &basic::rayleighQuotient<double, DenseObj<double>>, 
          "Compute the Rayleigh quotient for a given matrix and vector.",
          py::arg("A"), py::arg("b"));

    // Krylov Subspace
    m.def("arnoldi", &Krylov::Arnoldi<double, DenseObj<double>, VectorObj<double>>, 
          "Arnoldi iteration for generating an orthogonal basis of the Krylov subspace.",
          py::arg("A"), py::arg("Q"), py::arg("H"), py::arg("tol"));

    // Conjugate Gradient Solver
    py::class_<ConjugateGrad<double, SparseMatrixCSC<double>, VectorObj<double>>>(m, "ConjugateGrad")
        .def(py::init<const SparseMatrixCSC<double>&, const VectorObj<double>&, int, double>())
        .def("solve", &ConjugateGrad<double, SparseMatrixCSC<double>, VectorObj<double>>::solve,
             "Solve the linear system using the Conjugate Gradient method.");

    // Iterative Solvers
    py::class_<GradientDescent<double, SparseMatrixCSC<double>, VectorObj<double>>>(m, "GradientDescent")
        .def(py::init<const SparseMatrixCSC<double>&, const VectorObj<double>&, int, double>())
        .def("solve", &GradientDescent<double, SparseMatrixCSC<double>, VectorObj<double>>::solve,
             "Solve the linear system using Gradient Descent.");

    // MultiGrid Solver
    py::class_<AlgebraicMultiGrid<double, VectorObj<double>>>(m, "AlgebraicMultiGrid")
        .def(py::init<>())
        .def("amgVCycle", &AlgebraicMultiGrid<double, VectorObj<double>>::amgVCycle,
             "Perform one V-cycle of Algebraic MultiGrid.",
             py::arg("A"), py::arg("b"), py::arg("x"), py::arg("levels"), py::arg("smoothingSteps"), py::arg("theta"));

    // LU Factorization
    m.def("pivot_lu", &LU::PivotLU<double, DenseObj<double>>, 
        "Perform LU decomposition with partial pivoting.",
        py::arg("A"), py::arg("P"));

    // GMRES Solver
    py::class_<GMRES<double, SparseMatrixCSC<double>, VectorObj<double>>>(m, "GMRES")
        .def(py::init<>())
        .def("solve", &GMRES<double, SparseMatrixCSC<double>, VectorObj<double>>::solve,
             "Solve the linear system using GMRES.",
             py::arg("A"), py::arg("b"), py::arg("x"), py::arg("maxIter"), py::arg("KrylovDim"), py::arg("tol"));

    // Numerical Integration
   py::class_<GaussianQuadrature<double>>(m, "GaussQuadrature")
        .def(py::init<int>())
        .def("integrate", &GaussianQuadrature<double>::integrate, 
             "Perform numerical integration of a function over the interval [a, b].",
             py::arg("f"), py::arg("a"), py::arg("b"))
        .def("getPoints", &GaussianQuadrature<double>::getPoints, 
             "Get the quadrature points.")
        .def("getWeights", &GaussianQuadrature<double>::getWeights, 
             "Get the quadrature weights.");

    // ODE Solvers
    py::class_<RungeKutta<double>>(m, "RK4")
        .def(py::init<>())
        .def("solve", &RungeKutta<double, VectorObj<double>, VectorObj<double>>::solve,
             "Solve the ODE using the Runge-Kutta 4th order method.",
             py::arg("f"), py::arg("t0"), py::arg("y0"), py::arg("t_end"), py::arg("h"));

    // Matrix/Vector Operations
    py::class_<VectorObj<double>>(m, "Vector")
        .def(py::init<int>())
        .def("__getitem__", static_cast<double& (VectorObj<double>::*)(size_t)>(&VectorObj<double>::operator[]))
        .def("__getitem__", static_cast<const double& (VectorObj<double>::*)(size_t) const>(&VectorObj<double>::operator[]))
        .def("__setitem__", [](VectorObj<double>& v, size_t i, double val) { v[i] = val; })
        .def("size", &VectorObj<double>::size)
        .def("norm", &VectorObj<double>::L2norm, "Compute the L2 norm of the vector.");

    // Dense Matrix
    py::class_<DenseObj<double>>(m, "DenseMatrix")
        .def(py::init<int, int>())
        .def("__call__", static_cast<double& (DenseObj<double>::*)(int, int)>(&DenseObj<double>::operator()))
        .def("__call__", static_cast<const double& (DenseObj<double>::*)(int, int) const>(&DenseObj<double>::operator()))
        .def("rows", &DenseObj<double>::getRows)
        .def("cols", &DenseObj<double>::getCols);

    // Sparse Matrix
    py::class_<SparseMatrixCSC<double>>(m, "SparseMatrix")
        .def(py::init<int, int>())
        .def("addValue", &SparseMatrixCSC<double>::addValue)
        .def("finalize", &SparseMatrixCSC<double>::finalize)
        .def("rows", &SparseMatrixCSC<double>::getRows)
        .def("cols", &SparseMatrixCSC<double>::getCols);
}