// =============================
// Google Test Unit Tests
// =============================
#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <algorithm>
#include <random>
#include <chrono>
#include <cstdlib>
#include <ctime>

// Google Test header (make sure gtest is installed and include path is set)
#include <gtest/gtest.h>
#include "../src/LinearAlgebra/FastMultipole/FMM.hpp"

using namespace FASTSolver;

class FMMTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Set up common test data.
        domain_min = -1.0;
        domain_max = 1.0;
        // Correct ordering: (xmin, xmax, ymin, ymax)
        root = new FMM<double>::Cell(domain_min, domain_max, domain_min, domain_max);
    }

    void TearDown() override {
        delete root;
    }

    // Helper function to generate random particles.
    std::vector<FMM<double>::Particle*> generateRandomParticles(int n) {
        std::vector<FMM<double>::Particle*> particles;
        std::mt19937 gen(std::random_device{}());
        std::uniform_real_distribution<double> pos_dist(domain_min, domain_max);
        std::uniform_real_distribution<double> charge_dist(-1.0, 1.0);

        for (int i = 0; i < n; i++) {
            particles.push_back(new FMM<double>::Particle(
                pos_dist(gen), pos_dist(gen), charge_dist(gen)));
        }
        return particles;
    }

    double domain_min, domain_max;
    FMM<double>::Cell* root;
    FMM<double> fmm;
};

// Test particle creation and basic properties.
TEST_F(FMMTest, ParticleCreation) {
    auto particle = new FMM<double>::Particle(1.0, 2.0, 0.5);
    EXPECT_DOUBLE_EQ(particle->x, 1.0);
    EXPECT_DOUBLE_EQ(particle->y, 2.0);
    EXPECT_DOUBLE_EQ(particle->q, 0.5);
    delete particle;
}

// Test tree construction with few particles (should remain a leaf).
TEST_F(FMMTest, SimpleTreeConstruction) {
    auto particles = generateRandomParticles(5);
    root->particles = particles;
    
    auto tree = fmm.buildTree(root);
    EXPECT_TRUE(tree->is_leaf);
    EXPECT_EQ(tree->particles.size(), 5);
    EXPECT_TRUE(tree->children.empty());

    // Cleanup.
    for (auto p : particles)
        delete p;
}

// Test tree subdivision when number of particles exceeds MAX_PARTICLES.
TEST_F(FMMTest, TreeSubdivision) {
    auto particles = generateRandomParticles(20); // More than MAX_PARTICLES (10)
    root->particles = particles;
    
    auto tree = fmm.buildTree(root);
    EXPECT_FALSE(tree->is_leaf);
    EXPECT_TRUE(tree->particles.empty());
    EXPECT_EQ(tree->children.size(), 4);

    // Cleanup.
    for (auto p : particles)
        delete p;
}

// Test multipole expansion computation.
TEST_F(FMMTest, MultipoleExpansion) {
    // Create a simple configuration with a single particle.
    auto particle = new FMM<double>::Particle(0.1, 0.1, 1.0);
    root->particles.push_back(particle);
    
    fmm.computeLeafMultipole(root);
    
    // The monopole term should equal the total charge.
    EXPECT_DOUBLE_EQ(std::real(root->multipole[0]), 1.0);
    EXPECT_DOUBLE_EQ(std::imag(root->multipole[0]), 0.0);
    
    delete particle;
}

// Test the well-separated condition.
TEST_F(FMMTest, WellSeparatedCondition) {
    std::complex<double> far_point(10.0, 10.0);
    std::complex<double> near_point(1.1, 1.1);
    
    EXPECT_TRUE(fmm.wellSeparated(root, far_point));
    EXPECT_FALSE(fmm.wellSeparated(root, near_point));
}

// Test potential evaluation.
// The potential is computed using a 2D logarithmic kernel:
//   potential = - (1/(2π)) * sum(q_i * log(r_i))
TEST_F(FMMTest, PotentialEvaluation) {
    // Create two particles with unit charges.
    auto p1 = new FMM<double>::Particle(-0.5, -0.5, 1.0);
    auto p2 = new FMM<double>::Particle(0.5, 0.5, 1.0);
    root->particles = {p1, p2};
    
    // Evaluate potential at the origin.
    std::complex<double> eval_point(0.0, 0.0);
    double potential = fmm.evaluatePotential(root, eval_point);
    
    // Direct calculation:
    // For each particle, r = distance from eval_point.
    // p1: r1 = sqrt(0.5)  and p2: r2 = sqrt(0.5)
    // So, direct potential = -1/(2π) * [log(r1) + log(r2)]
    double r = std::sqrt(0.5);
    double direct_potential = - (std::log(r) + std::log(r)) / (2 * M_PI);
    EXPECT_NEAR(potential, direct_potential, 1e-10);
    
    delete p1;
    delete p2;
}

// Performance test: FMM setup (tree build and upward pass) should complete quickly.
TEST_F(FMMTest, PerformanceComparison) {
    const int N = 1000;
    auto particles = generateRandomParticles(N);
    root->particles = particles;
    
    // Build tree and compute multipole expansions.
    auto start = std::chrono::high_resolution_clock::now();
    auto tree = fmm.buildTree(root);
    fmm.upwardPass(tree);
    auto end = std::chrono::high_resolution_clock::now();
    auto fmm_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    // Cleanup.
    for (auto p : particles)
        delete p;
    
    // The test passes if FMM setup completes within 1 second.
    EXPECT_LT(fmm_time, 1000);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}