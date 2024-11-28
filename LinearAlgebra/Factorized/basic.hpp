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
    void gramSmith(const MatrixObj<TNum> &A, std::vector<VectorObj<TNum>> &orthSet_) {
        int m = A.get_col();

        for (int i = 0; i < m; i++) {
            VectorObj<TNum> v = A.get_Col(i);
            orthSet_[i] = v;
            for (int j = i-1; j > 0; j--) {
                orthSet_[j] = subtProj(orthSet_[j], v);
            }
        }
    }

    template <typename TNum>
    void genUnitvec(int i, int n, VectorObj<TNum> &e){
            TNum *unit_v = new TNum[n*1];
            memset(unit_v, 0, sizeof(TNum));
            unit_v[i] = static_cast<TNum>(1);
            e = VectorObj<TNum>(unit_v, n);
            if(unit_v!=nullptr){
                delete[] unit_v;
            }
    }

    template <typename TNum>
    void genUnitMatx(int n, int m, MatrixObj<TNum> &I){
        std::vector<VectorObj<TNum>> eSet(m);
        for(int i=0; i<n; i++){
            genUnitvec(i, n, eSet[i]);
        }
        I = MatrixObj<TNum>(eSet);
    }

    template<typename TNum> 
    TNum sign(TNum x){
        return x >= 0 ? static_cast<TNum>(1) : static_cast<TNum>(-1);
    }

    template <typename TNum>
    void houseH(MatrixObj<TNum> &A, MatrixObj<TNum> &H, int index){
        int m = A.get_col();
        int n = A.get_row();
        MatrixObj<TNum> I;
        genUnitMatx(n, m, I);

        VectorObj<TNum> x = A.get_Col(index);
        VectorObj<TNum> e;
        genUnitvec(index, n, e);
        VectorObj<TNum> vt = x + x.L2norm() * sign(x[index]) * e;
        MatrixObj<TNum> v(vt);
        H = I - (v * v.Transpose()) * (1/(vt * vt));
    }
}
#endif