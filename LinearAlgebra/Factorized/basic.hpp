#ifndef BASIC_HPP
#define BASIC_HPP

#include <iostream>
#include <vector>
#include "Obj/MatrixObj.hpp" 

namespace basic {

    template <typename TNum>
    void powerIter(MatrixObj<TNum> &A, VectorObj<TNum> &b, int max_iter_num) {
        while(max_iter_num--){
            b.normalized();
            MatrixObj<TNum> btemp(b);
            MatrixObj<TNum> Ab = A * btemp;
            b = Ab.get_Col(0);
        }
    }
    
    template <typename TNum>
    double rayleighQuotient(MatrixObj<TNum> &A, VectorObj<TNum> &b) {
        MatrixObj<TNum> btemp(b);
        MatrixObj<TNum> Ab = A * btemp;
        TNum dot_product = b * Ab.get_Col(0);
        return dot_product / (b * b);
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
    MatrixObj<TNum> gramSmith(MatrixObj<TNum> &A) {
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