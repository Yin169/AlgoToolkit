#include <gtest/gtest.h>
#include "GaussianQuad.hpp"
#include <cmath>

class GuassianQuadTest : public ::testing::Test {
protected:
    const double eps = 1e-8;
};

TEST_F(GuassianQuadTest, ConstructorTest) {
    EXPECT_NO_THROW(Quadrature::GaussianQuadrature<double>(5));
    EXPECT_THROW(Quadrature::GaussianQuadrature<double>(0), std::invalid_argument);
    EXPECT_THROW(Quadrature::GaussianQuadrature<double>(-1), std::invalid_argument);
}

TEST_F(GuassianQuadTest, PointsSymmetryTest) {
    Quadrature::GaussianQuadrature<double> quad(5);
    auto points = quad.getPoints();
    auto weights = quad.getWeights();
    
    EXPECT_EQ(points.size(), 5);
    EXPECT_EQ(weights.size(), 5);
    
    // Check symmetry of points
    for(size_t i = 0; i < points.size()/2; ++i) {
        EXPECT_NEAR(points[i], -points[points.size()-1-i], eps);
        EXPECT_NEAR(weights[i], weights[points.size()-1-i], eps);
    }
}

TEST_F(GuassianQuadTest, PolynomialIntegrationTest) {
    Quadrature::GaussianQuadrature<double> quad(5);
    
    // Integrate x^2 from -1 to 1
    auto f1 = [](double x) { return x*x; };
	
    EXPECT_NEAR(quad.integrate(f1, -1.0, 1.0), 2.0/3.0, eps);
    
    // Integrate x^3 from -1 to 1
    auto f2 = [](double x) { return x*x*x; };
    EXPECT_NEAR(quad.integrate(f2, -1.0, 1.0), 0.0, eps);
    
    // Integrate x^4 from -1 to 1
    auto f3 = [](double x) { return x*x*x*x; };
    EXPECT_NEAR(quad.integrate(f3, -1.0, 1.0), 2.0/5.0, eps);
}

TEST_F(GuassianQuadTest, TrigonometricIntegrationTest) {
    Quadrature::GaussianQuadrature<double> quad(10);
    std::vector<double> points = quad.getPoints();
    std::vector<double> weights = quad.getWeights();
    // Integrate sin(x) from 0 to pi
    auto f1 = [](double x) { return std::sin(x); };
    EXPECT_NEAR(quad.integrate(f1, 0.0, M_PI), 2.0, eps);
    
    // Integrate cos(x) from 0 to pi
    auto f2 = [](double x) { return std::cos(x); };
    EXPECT_NEAR(quad.integrate(f2, 0.0, M_PI), 0.0, eps);
}

TEST_F(GuassianQuadTest, AccuracyTest) {
    // Test with increasing number of points
    std::vector<int> points = {1, 2, 3, 4};
    auto f = [](double x) { return std::exp(x); };
    double exact = std::exp(1.0) - std::exp(-1.0);
    
    std::vector<double> errors;
    for(int n : points) {
        Quadrature::GaussianQuadrature<double> quad(n);
        double result = quad.integrate(f, -1.0, 1.0);
        errors.push_back(std::abs(result - exact));
    }
    
    // Check that errors decrease with more points
    for(size_t i = 1; i < errors.size(); ++i) {
        EXPECT_LT(errors[i], errors[i-1]);
    }
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}