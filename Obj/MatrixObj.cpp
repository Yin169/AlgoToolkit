#include <algorithm>
#include <cstring>

template<typename TObj>
class MatrixObj{
	private:
		int _n, _m;
		TObj *arr;
	public:
		MatrixObj(int n, int m) : _n(n), _m(m){
			arr = new TObj[ _n * _m];
		}

		~MatrixObj(){ delete[] arr; }

		int get_row(){return _n;}
		int get_col(){return _m;}

		MatrixObj(const MatrixObj &other){
			_n = other.get_row(), _m = other.get_col();
			arr = new TObj[_n, _m];
			std::copy(&other[0], &other[_n * _m], arr );
		}
		
		MatrixObj(const TObj *other, int n, int m){
			_n = n, _m = m;
			arr = new TObj[_n * _m];
			std::copy(other, other+ n * m, arr);
		}

		MatrixObj operator[](int n){return arr[n];}

		MatrixObj &operator=(const MatrixObj &other){
			if (&other != this){
				_n = other.get_row(), _m = other.get_col();
				arr = new TObj[_n, _m];
				std::copy(&other[0], &other[_n * _m], arr );	
			}
			return *this;
		}

		MatrixObj &operator+(const MatrixObj &other){
			if (_n==other.get_row() && _m==other.get_col()){
				for(int i=0;i<_n*_m; i++){arr[i]+=other[i];}
				return *this;
			}
			return nullptr;
		}
		MatrixObj &operator-(const MatrixObj &other){
			if (_n==other.get_row() && _m==other.get_col()){
				for(int i=0;i<_n*_m; i++){arr[i]-=other[i];}
				return *this;
			}
			return nullptr;
		}

		MatrixObj operator*(const MatrixObj &other){
			if (_m==other.get_row()){
				TObj *t = new TObj[_n * other.get_col()];
				memset(t, 0, sizeof(TObj) * _n * other.get_col());

				for (int i=0; i<_n; i++){
					for (int k=0; k<other.get_row(); k++){
						for(int j=0; j<other.get_col(); j++){
							t[i*_n+j] += arr[ i * _n + k ] * other[k * other.get_row() + j];
						}
					}
				}
				MatrixObj temp(t, _n, other.get_col());
				delete[] t;
				return temp;
			}
		}
};
