#ifndef BASIC_HPP
#define BASIC_HPP

#include <vector>
#include "../../Obj/MatrixObj.hpp" // 使用正确的相对路径

namespace basic {

    template <typename TNum>
    void powerIter(const MatrixObj<TNum> &A, VectorObj<TNum> &b, int max_iter_num) {
        while(max_iter_num--){
            b.normalized();
            VectorObj<TNum> Ab = A * b;
            b = Ab;
        }
    }

    template <typename TNum>
    double rayleighQuotient(const MatrixObj<TNum> &A, const VectorObj<TNum> &b) {
        VectorObj<TNum> Ab = A * b;
        return (b.Transpose() * Ab) / (b.Transpose() * b);
    }

    template <typename TNum>
    VectorObj<TNum> subtProj(const VectorObj<TNum> &u, const VectorObj<TNum> &v) {
        double factor = (u.Transpose() * v) / (v.Transpose * v);
        VectorObj<TNum> result(u.get_row(), u.get_col());
        for (int i = 0; i < u.get_row(); ++i) {
            result[i] = u[i] - factor * v[i];
        }
        return result;
    }

    template <typename TNum>
    MatrixObj<TNum> gramSmith(const MatrixObj<TNum> &A) {
        size_t n = A.get_row(), m = A.get_col();
        std::vector<VectorObj<TNum>> orthSet;

        for (size_t i = 0; i < m; i++) {
            VectorObj<TNum> u = A.get_Col(i);
            VectorObj<TNum> v(u);

            for (size_t j = 0; j < orthSet.size(); j++) {
                v = subtProj(v, orthSet[j]);
            }

            if (v.L2norm() > 0) { 
                orthSet.push_back(v);
            }
        }

        return MatrixObj<TNum>(orthSet); 
    }
}
