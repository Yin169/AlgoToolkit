#ifndef GAUSSIANQUAD_HPP
#define GAUSSIANQUAD_HPP

#include <vector>
#include <cmath>
#include <stdexcept>
#include <functional>

namespace Quadrature {

template<typename T>
class GaussianQuadrature {
private:
    std::vector<T> points;
    std::vector<T> weights;
    
    void computeLegendrePolynomial(int n, T x, T& Pn, T& Pn_minus_1) const {
        Pn = x;
        Pn_minus_1 = 1;

        for (int k = 2; k <= n; k++) {
            T Pn_minus_2 = Pn_minus_1;
            Pn_minus_1 = Pn;
            Pn = ((2.0 * k - 1.0) * x * Pn_minus_1 - (k - 1.0) * Pn_minus_2) / k;
        }
    }
    
    void generatePointsAndWeights(int n) {
        points.resize(n);
        weights.resize(n);
        
        const T eps = 1e-12;
        const int max_iter = 100;
        
        for (int i = 0; i < (n + 1) / 2; i++) {
            // Initial guess for the root
            T x = std::cos(M_PI * (i + 0.75) / (n + 0.5));
            T delta;
            
            // Newton's method iteration
            for (int iter = 0; iter < max_iter; iter++) {
                T Pn, Pn_minus_1;
                computeLegendrePolynomial(n, x, Pn, Pn_minus_1);
                
                // First derivative of Legendre polynomial
                T Pn_prime = n * (x * Pn - Pn_minus_1) / (x * x - 1.0);
                
                // Newton's method update
                delta = -Pn / Pn_prime;
                x += delta;
                
                if (std::abs(delta) <= eps) break;
            }
            T Pn, Pn_minus_1;
            computeLegendrePolynomial(n, x, Pn, Pn_minus_1);
            
            // Compute weights
            T Pn_prime = n * (x * Pn - Pn_minus_1) / (x * x - 1.0);
            T w = 2.0 / ((1.0 - x * x) * Pn_prime * Pn_prime);
            
            points[i] = -x;
            points[n-1-i] = x;
            weights[i] = w;
            weights[n-1-i] = w;
        }
    }

public:
    GaussianQuadrature(int n) {
        if (n < 1) throw std::invalid_argument("Number of points must be positive");
        generatePointsAndWeights(n);
    }
    
    T integrate(const std::function<T(T)>& f, T a, T b) const {
        T sum = 0.0;
        T mid = (b + a) / 2.0;
        T scale = (b - a) / 2.0;
        
        for (size_t i = 0; i < points.size(); i++) {
            T x = mid + scale * points[i];
            sum += weights[i] * f(x);
        }
        
        return scale * sum;
    }
    
    const std::vector<T>& getPoints() const { return points; }
    const std::vector<T>& getWeights() const { return weights; }
};

} // namespace Quadrature

#endif // GAUSSIANQUAD_HPP