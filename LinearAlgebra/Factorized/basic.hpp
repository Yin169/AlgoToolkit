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
        VectorObj<TNum> result(u.get_row());
        for (int i = 0; i < u.get_row(); ++i) {
            result[i] = u[i] - factor * v[i];
        }
        return result;
    }

    template <typename TNum>
    void gramSmith(const MatrixObj<TNum> &A, MatrixObj<TNum> &orth_) {
        int m = A.get_col();
        std::vector<VectorObj<TNum>> orthSet(m);

        for (int i = 0; i < m; i++) {
            VectorObj<TNum> v = A.get_Col(i);
            for (int j = i+1; j < m; j++) {
                VectorObj<TNum> u = A.get_Col(j);
                v = subtProj(u, v);
            }
            orthSet[i] = v;
        }
        MatrixObj<TNum> res(orthSet);
        orth_ = res;
    }
}
#endif