#ifndef BASIC_HPP
#define BASIC_HPP

#include <iostream>
#include <vector>
#include "Obj/MatrixObj.hpp" 

namespace basic {

    template <typename TNum>
    void powerIter(const MatrixObj<TNum> &A, VectorObj<TNum> &b, int max_iter_num) {
        while(max_iter_num--){
            b.normalized();
            MatrixObj<TNum> Atemp = A;
            MatrixObj<TNum> btemp = static_cast<MatrixObj<TNum>>(b);
            MatrixObj<TNum> Ab = Atemp * btemp;
            b = Ab.get_Col(0);
        }
    }
    
    template <typename TNum>
    double rayleighQuotient(const MatrixObj<TNum> &A, const VectorObj<TNum> &b) {
        VectorObj<TNum> btemp = b;
        MatrixObj<TNum> bcast = static_cast<MatrixObj<TNum>>(btemp);
        MatrixObj<TNum> Atemp = A;
        MatrixObj<TNum> Ab = Atemp * bcast;
        TNum dot_product = btemp * Ab.get_Col(0);
        return dot_product / (btemp * btemp);
    }

    template <typename TNum>
    VectorObj<TNum> subtProj(VectorObj<TNum> &u, VectorObj<TNum> &v) {
        double factor = (u * v) / (v * v);
        VectorObj<TNum> result(u.get_col());
        for (int i = 0; i < u.get_col(); ++i) {
            result[i] = u[i] - factor * v[i];
        }
        return result;
    }

    template <typename TNum>
    MatrixObj<TNum> gramSmith(const MatrixObj<TNum> &A) {
        size_t m = A.get_col();
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
#endif