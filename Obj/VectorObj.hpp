#ifndef VECTOROBJ_HPP
#define VECTOROBJ_HPP

#include "./MatrixObj.hpp"

template <typename TObj>
class VectorObj : private MatrixObj<TObj> {
	public:
		VectorObj(const TObj *other, int n) : MatrixObj<TObj>(other, n, 1){}
		~VectorObj(){}

		double L2norm(){
			double sum=0;
			for (int i=0; i<MatrixObj<TObj>::get_row() * MatrixObj<TObj>::get_col(); i++){
				sum += MatrixObj<TObj>::arr[i] * MatrixObj<TObj>::arr[i];
			}
			return std::sqrt(sum);
		}

		void normalized(){
			double l2norm = L2norm();
			for (int i=0; i<MatrixObj<TObj>::get_row(), MatrixObj<TObj>::get_col(); i++){
				MatrixObj<TObj>::arr[i] /= l2norm; 
			}
		} 
};

#endif
