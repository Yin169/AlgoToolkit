#include <gtest/gtest.h>
#include "../src/Regression/linearRegress.hpp"
#include "../src/Obj/DenseObj.hpp"
#include <cmath>

class LinearRegressTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Simple dataset: y = 2x₁ + 3x₂ + 1
        x_train = DenseObj<double>(4, 2);
        y_train = VectorObj<double>(4);
        
        // Training data
        x_train(0,0) = 1.0; x_train(0,1) = 2.0;  // y = 8
        x_train(1,0) = 2.0; x_train(1,1) = 1.0;  // y = 7
        x_train(2,0) = 3.0; x_train(2,1) = 3.0;  // y = 13
        x_train(3,0) = 4.0; x_train(3,1) = 2.0;  // y = 11
        
        y_train[0] = 9.0;
        y_train[1] = 8.0;
        y_train[2] = 14.0;
        y_train[3] = 12.0;
        
        model = new LinearRegress<double, DenseObj<double>>(2);
    }

    void TearDown() override {
        delete model;
    }

    DenseObj<double> x_train;
    VectorObj<double> y_train;
    LinearRegress<double, DenseObj<double>>* model;
};

TEST_F(LinearRegressTest, ConstructorTest) {
    LinearRegress<double, DenseObj<double>> regressor(2);
    EXPECT_NO_THROW(regressor);
}

TEST_F(LinearRegressTest, FitTest) {
    EXPECT_NO_THROW(model->fit(x_train, y_train));
}

TEST_F(LinearRegressTest, PredictionTest) {
    model->fit(x_train, y_train);
    
    VectorObj<double> test_input(2);
    test_input[0] = 2.0;
    test_input[1] = 2.0;
    
    double prediction = model->predict(test_input);
    // Expected value should be close to: 2*2 + 3*2 + 1 = 11
    EXPECT_NEAR(prediction, 11.0, 1);
}

TEST_F(LinearRegressTest, InvalidInputTest) {
    VectorObj<double> invalid_y(3); // Wrong size
    EXPECT_THROW(model->fit(x_train, invalid_y), std::invalid_argument);
}

TEST_F(LinearRegressTest, ToleranceTest) {
    EXPECT_NO_THROW(model->fit(x_train, y_train, 1000, 1e-10));
}

TEST_F(LinearRegressTest, MaxIterationsTest) {
    EXPECT_NO_THROW(model->fit(x_train, y_train, 10, 1e-8));
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}