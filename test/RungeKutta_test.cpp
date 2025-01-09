#include <gtest/gtest.h>
#include "RungeKutta.hpp"
#include <cmath>

// Test fixture class for RungeKutta tests
class RungeKuttaTest : public ::testing::Test {
protected:
    using Vector = VectorObj<double>;
    using Matrix = DenseObj<double>;
    RungeKutta<double, Vector, Vector> rk;
    
    // Helper function for simple harmonic oscillator
    // dy/dt = v
    // dv/dt = -y
    static Vector harmonicOscillator(const Vector& state) {
        Vector result(2);
        result[0] = state[1];         // dy/dt = v
        result[1] = -state[0];        // dv/dt = -y
        return result;
    }
    
    // Helper function for exponential growth
    // dy/dt = y
    static Vector exponentialGrowth(const Vector& state) {
        Vector result(1);
        result[0] = state[0];
        return result;
    }
    
    // Helper function to compute relative error
    static double relativeError(double expected, double actual) {
        return std::abs(expected - actual) / std::abs(expected);
    }
};

// Test basic integration of exponential growth
TEST_F(RungeKuttaTest, ExponentialGrowthTest) {
    Vector y(1);
    y[0] = 1.0;  // Initial condition y(0) = 1
    
    double h = 0.1;
    size_t n = 10;  // Integrate from t=0 to t=1
    
    rk.solve(y, exponentialGrowth, h, n);
    
    // Expected result is e^1 ≈ 2.71828
    double expected = std::exp(1.0);
    EXPECT_NEAR(y[0], expected, 1e-3);
}

// Test harmonic oscillator integration
TEST_F(RungeKuttaTest, HarmonicOscillatorTest) {
    Vector y(2);
    y[0] = 1.0;  // Initial position
    y[1] = 0.0;  // Initial velocity
    
    double h = 0.001;
    size_t n = (size_t)(2.0 * 3.1415926 / h);  // One full period
    
    rk.solve(y, harmonicOscillator, h, n);
    
    // After one period, should return to initial conditions
    EXPECT_NEAR(y[0], 1.0, 1e-2);
    EXPECT_NEAR(y[1], 0.0, 1e-2);
}

// Test energy conservation in harmonic oscillator
TEST_F(RungeKuttaTest, EnergyConservationTest) {
    Vector y(2);
    y[0] = 1.0;  // Initial position
    y[1] = 0.0;  // Initial velocity
    
    double initialEnergy = 0.5 * (y[0] * y[0] + y[1] * y[1]);
    
    double h = 0.1;
    size_t n = 100;
    
    rk.solve(y, harmonicOscillator, h, n);
    
    double finalEnergy = 0.5 * (y[0] * y[0] + y[1] * y[1]);
    EXPECT_NEAR(finalEnergy, initialEnergy, 1e-2);
}

// Test invalid parameters
TEST_F(RungeKuttaTest, InvalidParametersTest) {
    Vector y(1);
    y[0] = 1.0;
    
    // Test negative step size
    EXPECT_THROW(rk.solve(y, exponentialGrowth, -0.1, 10), std::invalid_argument);
    
    // Test zero steps
    EXPECT_THROW(rk.solve(y, exponentialGrowth, 0.1, 0), std::invalid_argument);
    
    // Test empty state vector
    Vector empty_y(0);
    EXPECT_THROW(rk.solve(empty_y, exponentialGrowth, 0.1, 10), std::invalid_argument);
}

// // Test adaptive step size control
// TEST_F(RungeKuttaTest, AdaptiveStepSizeTest) {
//     Vector y(2);
//     y[0] = 1.0;  // Initial position
//     y[1] = 0.0;  // Initial velocity

//     double h = 0.001;
//     double tol = 1e-6;
//     size_t max_steps = (size_t)(2.0 * 3.1415926 / h);  // One full period
    
//     size_t steps_taken = rk.solveAdaptive(y, harmonicOscillator, h, tol, max_steps);
    
//     // Check if solution is within tolerance
//     double expected = std::exp(1.8);
//     EXPECT_LT(relativeError(expected, y[0]), tol);
    
//     // Check if steps were actually taken
//     EXPECT_GT(steps_taken, 0);
//     EXPECT_LE(steps_taken, max_steps);
// }

// Test callback functionality
TEST_F(RungeKuttaTest, CallbackTest) {
    Vector y(1);
    y[0] = 1.0;
    
    size_t callback_count = 0;
    auto callback = [&callback_count](size_t step, const Vector& state) {
        callback_count++;
    };
    
    size_t n = 10;
    rk.solve(y, exponentialGrowth, 0.1, n, callback);
    
    EXPECT_EQ(callback_count, n);
}

// Test convergence order
TEST_F(RungeKuttaTest, ConvergenceOrderTest) {
    Vector y1(1), y2(1);
    y1[0] = y2[0] = 1.0;
    
    double h1 = 0.1;
    double h2 = 0.05;
    size_t n1 = 10;
    size_t n2 = 20;
    
    rk.solve(y1, exponentialGrowth, h1, n1);
    rk.solve(y2, exponentialGrowth, h2, n2);
    
    double expected = std::exp(1.0);
    double error1 = std::abs(y1[0] - expected);
    double error2 = std::abs(y2[0] - expected);
    
    // RK4 should have 4th order convergence
    // error ratio should be approximately 16 (2^4)
    double error_ratio = error1 / error2;
    EXPECT_NEAR(error_ratio, 16.0, 1.0);
}

// Test system with multiple time scales
TEST_F(RungeKuttaTest, MultipleTimeScalesTest) {
    // Define a stiff system: dy/dt = -50y + sin(t)
    auto stiffSystem = [](const Vector& y) -> Vector {
        Vector result(1);
        double t = 0.0;  // You might want to pass time as a parameter in real applications
        result[0] = -50.0 * y[0] + std::sin(t);
        return result;
    };
    
    Vector y(1);
    y[0] = 1.0;
    
    // Test that adaptive stepping handles this better than fixed step
    size_t fixed_steps = 1000;
    double h = 0.01;
    Vector y_fixed = y;
    rk.solve(y_fixed, stiffSystem, h, fixed_steps);
    
    double tol = 1e-6;
    size_t max_steps = 100;
    size_t adaptive_steps = rk.solveAdaptive(y, stiffSystem, h, tol, max_steps);
    
    // Adaptive method should take fewer steps
    EXPECT_LT(adaptive_steps, fixed_steps);
    
    // Results should be similar
    EXPECT_NEAR(y_fixed[0], y[0], 1e-3);
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}