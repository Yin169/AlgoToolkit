#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>  // 确保包含 functional.h
#include "../src/Obj/SparseObj.hpp"
#include "../src/Obj/DenseObj.hpp"
#include "../src/Obj/VectorObj.hpp"
#include "../src/Intergal/GaussianQuad.hpp"
#include "../src/ODE/RungeKutta.hpp"
#include "../src/LinearAlgebra/Preconditioner/ILU.hpp"
#include "../src/LinearAlgebra/Preconditioner/MultiGrid.hpp"
#include "../src/LinearAlgebra/Krylov/ConjugateGradient.hpp"
#include "../src/LinearAlgebra/Krylov/GMRES.hpp"
#include "../src/LinearAlgebra/Krylov/KrylovSubspace.hpp"
#include "../src/LinearAlgebra/Solver/IterSolver.hpp"
#include "../src/LinearAlgebra/Factorized/basic.hpp"
#include "../src/utils.hpp"
#include <pybind11/numpy.h>

using namespace Quadrature;
namespace py = pybind11;

// 自定义异常
class fastsolverError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

PYBIND11_MODULE(fastsolver, m) {
    m.doc() = "Pybind11 interface for numerical solvers and linear algebra tools";

    // 注册自定义异常
    py::register_exception<fastsolverError>(m, "fastsolverError");

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

    // Matrix-Vector Multiplication
    m.def("matvec_mul", [](const SparseMatrixCSC<double> &A, const VectorObj<double> &x) {
        return A * x;
    }, "Matrix-vector multiplication", py::arg("A"), py::arg("x"));

    // Matrix-Matrix Multiplication
    m.def("matmat_mul", [](const SparseMatrixCSC<double> &A, const SparseMatrixCSC<double> &B) {
        return A * B;
    }, "Matrix-matrix multiplication", py::arg("A"), py::arg("B"));

    // Conjugate Gradient Solver
    py::class_<ConjugateGrad<double, SparseMatrixCSC<double>, VectorObj<double>>>(m, "ConjugateGrad")
        .def(py::init<const SparseMatrixCSC<double>&, const VectorObj<double>&, int, double>(),
            py::arg("A"), py::arg("b"), py::arg("max_iter"), py::arg("tol"))
        .def("solve", &ConjugateGrad<double, SparseMatrixCSC<double>, VectorObj<double>>::solve,
             "Solve the linear system using the Conjugate Gradient method.",py::arg("x"));

    // Iterative Solvers
    py::class_<GradientDescent<double, SparseMatrixCSC<double>, VectorObj<double>>>(m, "GradientDescent")
        .def(py::init<const SparseMatrixCSC<double>&, const VectorObj<double>&, int, double>(),
              py::arg("A"), py::arg("b"), py::arg("max_iter"), py::arg("tol"))
        .def("solve", &GradientDescent<double, SparseMatrixCSC<double>, VectorObj<double>>::solve,
             "Solve the linear system using Gradient Descent.", py::arg("x"));

    // MultiGrid Solver
    py::class_<AlgebraicMultiGrid<double, VectorObj<double>>>(m, "AlgebraicMultiGrid")
        .def(py::init<>())
        .def("amgVCycle", &AlgebraicMultiGrid<double, VectorObj<double>>::amgVCycle,
             "Perform one V-cycle of Algebraic MultiGrid.",
             py::arg("A"), py::arg("b"), py::arg("x"), py::arg("levels"), py::arg("smoothingSteps"), py::arg("theta"));

    // LU Factorization
    m.def("pivot_lu", &basic::PivotLU<double, DenseObj<double>>,
        "Perform LU decomposition with partial pivoting.",
        py::arg("A"), py::arg("P"));

    // GMRES Solver
    py::class_<GMRES<double, SparseMatrixCSC<double>, VectorObj<double>>>(m, "GMRES")
        .def(py::init<>())
        .def("enablePreconditioner", &GMRES<double, SparseMatrixCSC<double>, VectorObj<double>>::enablePreconditioner,
             "Enable preconditioning for GMRES.")
        .def("solve", &GMRES<double, SparseMatrixCSC<double>, VectorObj<double>>::solve,
             "Solve the linear system using GMRES.",
             py::arg("A"), py::arg("b"), py::arg("x"), py::arg("maxIter"), py::arg("KrylovDim"), py::arg("tol"));

    // Numerical Integration
   py::class_<GaussianQuadrature<double>>(m, "GaussQuadrature")
        .def(py::init<int>(), py::arg("n"))
        .def("integrate", &GaussianQuadrature<double>::integrate,
             "Perform numerical integration of a function over the interval [a, b].",
             py::arg("f"), py::arg("a"), py::arg("b"))
        .def("getPoints", &GaussianQuadrature<double>::getPoints,
             "Get the quadrature points.")
        .def("getWeights", &GaussianQuadrature<double>::getWeights,
             "Get the quadrature weights.");

    py::class_<RungeKutta<double, VectorObj<double>, VectorObj<double>>>(m, "RK4")
        .def(py::init<>())
        .def("solve", &RungeKutta<double, VectorObj<double>, VectorObj<double>>::solve,
            py::arg("y"), py::arg("f"), py::arg("h"), py::arg("n"), py::arg("callback") = nullptr);
        // .def("solve_adaptive", &RungeKutta<double, VectorObj<double>, VectorObj<double>>::solveAdaptive,
        //     py::arg("y"), py::arg("f"), py::arg("h"), py::arg("tol"), py::arg("max_steps"));

    // Matrix/Vector Operations
    // Update Vector bindings with numpy compatibility
    py::class_<VectorObj<double>>(m, "Vector")
        .def(py::init<int>(), py::arg("size"))
        .def(py::init([](py::array_t<double> array) {
            auto buf = array.request();
            if (buf.ndim != 1) {
                throw std::runtime_error("Number of dimensions must be 1");
            }
            VectorObj<double> vec(buf.shape[0]);
            std::memcpy(vec.element(), buf.ptr, buf.shape[0] * sizeof(double));
            return vec;
        }))
        .def("__array__", [](const VectorObj<double> &vec) {
            return py::array_t<double>({vec.size()}, {sizeof(double)}, vec.element());
        })
        .def("to_numpy", [](const VectorObj<double> &vec) {
            return py::array_t<double>({vec.size()}, {sizeof(double)}, vec.element());
        })
        .def("__getitem__", [](const VectorObj<double> &self, size_t index) -> double {
            if (index >= self.size()) {
                throw py::index_error("Index out of range");
            }
            return self[index];
        })
        .def("__setitem__", [](VectorObj<double> &self, size_t index, double value) {
            if (index >= self.size()) {
                throw py::index_error("Index out of range");
            }
            self[index] = value;
        })
        .def("size", &VectorObj<double>::size, "Get the size of the vector.")
        .def("norm", &VectorObj<double>::L2norm, "Compute the L2 norm of the vector.");

    // Add Cholesky decomposition
    m.def("cholesky", &basic::Cholesky<double, DenseObj<double>>,
          "Perform Cholesky decomposition of a symmetric positive definite matrix",
          py::arg("A"), py::arg("L"));

    // Update Dense Matrix bindings
    py::class_<DenseObj<double>>(m, "DenseMatrix")
        .def(py::init<int, int>(), py::arg("rows"), py::arg("cols"))
        .def(py::init<const VectorObj<double>&, int, int>(), py::arg("data"), py::arg("rows"), py::arg("cols"))
        .def(py::init<const std::vector<VectorObj<double>>&, int, int>(), py::arg("data"), py::arg("rows"), py::arg("cols"))
        .def("__setitem__", [](DenseObj<double> &self, std::pair<int, int> idx, double value) {
            self(idx.first, idx.second) = value;
        })
        .def("__getitem__", [](const DenseObj<double> &self, std::pair<int, int> idx) {
            return self(idx.first, idx.second);
        })
        .def("rows", &DenseObj<double>::getRows, "Get the number of rows.")
        .def("cols", &DenseObj<double>::getCols, "Get the number of columns.")
        .def("__len__", &DenseObj<double>::getRows, "Get the number of rows.")
        .def("resize", &DenseObj<double>::resize, py::arg("rows"), py::arg("cols"))
        .def("transpose", &DenseObj<double>::Transpose, "Get the transpose of the matrix.")
        .def("__mul__", [](const DenseObj<double>& self, const DenseObj<double>& other) {
            return self * other;
        }, py::is_operator())
        .def("__mul__", [](const DenseObj<double>& self, double scalar) {
            return self * scalar;
        }, py::is_operator());

    // Update Sparse Matrix bindings
    py::class_<SparseMatrixCSC<double>>(m, "SparseMatrix")
        .def(py::init<int, int>(), py::arg("rows"), py::arg("cols"))
        .def(py::init<const DenseObj<double>&>(), py::arg("dense_matrix"))
        .def("addValue", &SparseMatrixCSC<double>::addValue, py::arg("row"), py::arg("col"), py::arg("value"))
        .def("finalize", &SparseMatrixCSC<double>::finalize, "Finalize the sparse matrix construction.")
        .def("rows", &SparseMatrixCSC<double>::getRows, "Get the number of rows.")
        .def("cols", &SparseMatrixCSC<double>::getCols, "Get the number of columns.")
        .def("transpose", &SparseMatrixCSC<double>::Transpose, "Get the transpose of the matrix.")
        .def("__mul__", [](const SparseMatrixCSC<double>& self, const VectorObj<double>& vec) {
            return self * vec;
        }, py::is_operator())
        .def("__mul__", [](const SparseMatrixCSC<double>& self, const SparseMatrixCSC<double>& other) {
            return self * other;
        }, py::is_operator())
        .def("__mul__", [](const SparseMatrixCSC<double>& self, double scalar) {
            return self * scalar;
        }, py::is_operator())
        .def("__call__", [](const SparseMatrixCSC<double>& self, int i, int j) {
            return self(i, j);
        },py::arg("i"),py::arg("j"));

    // Matrix Market File IO
    m.def("read_matrix_market", [](const std::string& filename, py::object matrix) {
        if (py::isinstance<DenseObj<double>>(matrix)) {
            auto& mat = matrix.cast<DenseObj<double>&>();
            utils::readMatrixMarket<double, DenseObj<double>>(filename, mat);
        } else if (py::isinstance<SparseMatrixCSC<double>>(matrix)) {
            auto& mat = matrix.cast<SparseMatrixCSC<double>&>();
            utils::readMatrixMarket<double, SparseMatrixCSC<double>>(filename, mat);
        } else {
            throw fastsolverError("Unsupported matrix type");
        }
    }, "Read matrix from Matrix Market format file",
       py::arg("filename"), py::arg("matrix"));
}
