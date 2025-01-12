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

namespace py = pybind11;

PYBIND11_MODULE(numerical_solver, m) {
    m.doc() = "Pybind11 interface for numerical solvers and linear algebra tools";

    // Basic Operations
    m.def("power_iter", &basic::powerIter<double, DenseObj<double>>, "Power iteration method",
          py::arg("A"), py::arg("b"), py::arg("max_iter"));
    m.def("rayleigh_quotient", &basic::rayleighQuotient<double, DenseObj<double>>, "Compute Rayleigh quotient",
          py::arg("A"), py::arg("b"));

    // Krylov Subspace
    m.def("arnoldi", &Krylov::Arnoldi<double, DenseObj<double>, VectorObj<double>>, "Arnoldi iteration for Krylov subspace",
          py::arg("A"), py::arg("Q"), py::arg("H"), py::arg("tol"));

    // Conjugate Gradient Solver
    py::class_<ConjugateGrad<double, SparseMatrixCSC<double>, VectorObj<double>>>(m, "ConjugateGrad")
        .def(py::init<const SparseMatrixCSC<double>&, const VectorObj<double>&, int, double>())
        .def("solve", &ConjugateGrad<double, SparseMatrixCSC<double>, VectorObj<double>>::solve);

    // Iterative Solvers
    py::class_<GradientDescent<double, SparseMatrixCSC<double>, VectorObj<double>>>(m, "GradientDescent")
        .def(py::init<const SparseMatrixCSC<double>&, const VectorObj<double>&, int, double>())
        .def("solve", &GradientDescent<double, SparseMatrixCSC<double>, VectorObj<double>>::solve);

    py::class_<StaticIterMethod<double, SparseMatrixCSC<double>, VectorObj<double>>>(m, "StaticIterMethod")
        .def(py::init<const SparseMatrixCSC<double>&, const VectorObj<double>&, int, double>())
        .def("solve", &StaticIterMethod<double, SparseMatrixCSC<double>, VectorObj<double>>::solve);

    // MultiGrid Solver
    py::class_<AlgebraicMultiGrid<double, VectorObj<double>>>(m, "AlgebraicMultiGrid")
        .def(py::init<>())
        .def("amgVCycle", &AlgebraicMultiGrid<double, VectorObj<double>>::amgVCycle,
             py::arg("A"), py::arg("b"), py::arg("x"), py::arg("levels"), py::arg("smoothingSteps"), py::arg("theta"));

    // LU Factorization
    py::class_<LU<double, DenseObj<double>>>(m, "LU")
        .def(py::init<const DenseObj<double>&>())
        .def("solve", &LU<double, DenseObj<double>>::solve)
        .def("det", &LU<double, DenseObj<double>>::det);

    // GMRES Solver
    py::class_<GMRES<double, SparseMatrixCSC<double>, VectorObj<double>>>(m, "GMRES")
        .def(py::init<>())
        .def("solve", &GMRES<double, SparseMatrixCSC<double>, VectorObj<double>>::solve,
             py::arg("A"), py::arg("b"), py::arg("x"), py::arg("maxIter"), 
             py::arg("KrylovDim"), py::arg("tol"));

    // Numerical Integration
    py::class_<Quadrature::GaussianQuadrature<double>>(m, "GaussQuadrature")
        .def(py::init<int>())
        .def("integrate", &Quadrature::GaussianQuadrature<double>::integrate)
        .def("getPoints", &Quadrature::GaussianQuadrature<double>::getPoints)
        .def("getWeights", &Quadrature::GaussianQuadrature<double>::getWeights);

    // ODE Solvers
    py::class_<RungeKutta::RK4<double>>(m, "RK4")
        .def(py::init<>())
        .def("solve", &RungeKutta::RK4<double>::solve,
             py::arg("f"), py::arg("t0"), py::arg("y0"), 
             py::arg("t_end"), py::arg("h"));

    // LU Decomposition
    m.def("lu_decompose", &LU::luDecomposition<double, DenseObj<double>>, 
          "LU decomposition of matrix",
          py::arg("A"), py::arg("L"), py::arg("U"));

    m.def("lu_solve", &LU::luSolve<double, DenseObj<double>>, 
          "Solve using LU decomposition",
          py::arg("L"), py::arg("U"), py::arg("b"));

    // Matrix/Vector Operations
    py::class_<VectorObj<double>>(m, "Vector")
        .def(py::init<int>())
        .def("__getitem__", &VectorObj<double>::operator[])
        .def("__setitem__", [](VectorObj<double>& v, size_t i, double val) { v[i] = val; })
        .def("size", &VectorObj<double>::size)
        .def("norm", &VectorObj<double>::L2norm);

    py::class_<DenseObj<double>>(m, "DenseMatrix")
        .def(py::init<int, int>())
        .def("__call__", &DenseObj<double>::operator())
        .def("rows", &DenseObj<double>::getRows)
        .def("cols", &DenseObj<double>::getCols);

    py::class_<SparseMatrixCSC<double>>(m, "SparseMatrix")
        .def(py::init<int, int>())
        .def("addValue", &SparseMatrixCSC<double>::addValue)
        .def("finalize", &SparseMatrixCSC<double>::finalize)
        .def("rows", &SparseMatrixCSC<double>::getRows)
        .def("cols", &SparseMatrixCSC<double>::getCols);
}