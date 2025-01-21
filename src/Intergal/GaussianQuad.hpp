#ifndef GAUSSIANQUAD_HPP
#define GAUSSIANQUAD_HPP

#include <vector>
#include <cmath>
#include <stdexcept>
#include <functional>
#include <limits>

namespace Quadrature {

template<typename T>
class GaussianQuadrature {
private:
    std::vector<T> points;
    std::vector<T> weights;
        
    void computeLegendrePolynomial(int n, T x, T& Pn, T& Pn_minus_1) const {
        if (std::abs(x) > 1.0 + std::numeric_limits<T>::epsilon()) {
            throw std::domain_error("Argument x must be in [-1,1]");
        }

        Pn = x;
        Pn_minus_1 = 1;

        for (int k = 2; k <= n; k++) {
            T Pn_minus_2 = Pn_minus_1;
            Pn_minus_1 = Pn;
            // Use Bonnet's recursion formula for better numerical stability
            Pn = ((2 * k - 1) * x * Pn_minus_1 - (k - 1) * Pn_minus_2) / k;
        }
    }

    void generatePointsAndWeights(int n) {
        points.resize(n);
        weights.resize(n);
        
        const T eps = std::numeric_limits<T>::epsilon();
        const int max_iter = 100;
        
        for (int i = 0; i < (n + 1) / 2; i++) {
            // Initial guess using asymptotic formula
            T x = std::cos(M_PI * (4 * i + 3) / (4 * n + 2));
            T delta;
            
            // Newton's method iteration
            int iter;
            for (iter = 0; iter < max_iter; iter++) {
                T Pn, Pn_minus_1;
                computeLegendrePolynomial(n, x, Pn, Pn_minus_1);
                
                // First derivative of Legendre polynomial
                T Pn_prime = n * (x * Pn - Pn_minus_1) / (x * x - 1.0);
                
                // Newton's method update
                delta = -Pn / Pn_prime;
                x += delta;
                
                if (std::abs(delta) <= eps * std::abs(x)) break;
            }

            if (iter >= max_iter) {
                throw std::runtime_error("Failed to converge in computing Gauss points");
            }

            T Pn, Pn_minus_1;
            computeLegendrePolynomial(n, x, Pn, Pn_minus_1);
            
            // Compute weights using the derivative formula
            T Pn_prime = n * (x * Pn - Pn_minus_1) / (x * x - 1.0);
            T w = 2.0 / ((1.0 - x * x) * Pn_prime * Pn_prime);
            
            points[i] = -x;
            points[n-1-i] = x;
            weights[i] = w;
            weights[n-1-i] = w;
        }

        // Handle middle point for odd number of points
        if (n % 2 == 1) {
            T x = 0;
            T Pn, Pn_minus_1;
            computeLegendrePolynomial(n, x, Pn, Pn_minus_1);
            T Pn_prime = n * (x * Pn - Pn_minus_1) / (x * x - 1.0);
            points[n/2] = x;
            weights[n/2] = 2.0 / (Pn_prime * Pn_prime);
        }
    }

public:
    /**
     * @brief Constructs a Gaussian quadrature integrator
     * @param n Number of quadrature points
     * @throws std::invalid_argument if n < 1
     */
    explicit GaussianQuadrature(int n) {
        if (n < 1) throw std::invalid_argument("Number of points must be positive");
        generatePointsAndWeights(n);
    }
    
    /**
     * @brief Integrates a function over [a,b]
     * @param f Function to integrate
     * @param a Lower bound
     * @param b Upper bound
     * @return Approximate integral value
     * @throws std::invalid_argument if b <= a
     */
    T integrate(const std::function<T(T)>& f, T a, T b) const {
        if (b <= a) {
            throw std::invalid_argument("Upper bound must be greater than lower bound");
        }

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