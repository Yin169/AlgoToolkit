

#include <algorithm>
#include "../../Obj/MatrixObj.hpp"
#include "../../Obj/VectorObj.hpp"

template <typename TNum>
class IterSolverBase{
	public:
		SolverBase(){}
		~SolverBase(){}

		virtual itrFunc()=0;
		virtual TNum CalGradient()=0;
};
