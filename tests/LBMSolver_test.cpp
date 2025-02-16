#include <gtest/gtest.h>
#include "LBMSolver.hpp"

class LBMSolverTest : public ::testing::Test {
protected:
    std::array<size_t, 2> dims2D = {3, 3};
    std::array<size_t, 3> dims3D = {3, 3, 3};
    std::unique_ptr<MeshObj<double, 2>> mesh2D;
    std::unique_ptr<LBMSolver<double, 2>> solver2D;
    
    void SetUp() override {
        mesh2D = std::make_unique<MeshObj<double, 2>>(dims2D, 1.0);
        solver2D = std::make_unique<LBMSolver<double, 2>>(*mesh2D, 0.1);
    }
    
    bool isClose(double a, double b, double tol = 1e-10) {
        return std::abs(a - b) < tol;
    }
    
    double computeTotalMass() {
        double mass = 0.0;
        const auto& nodes = mesh2D->getNodes();
        for(const auto& node : nodes) {
            if(node.isActive) {
                mass += std::accumulate(node.distributions.begin(), 
                                     node.distributions.end(), 0.0);
            }
        }
        return mass;
    }
    
    std::array<double, 2> computeTotalMomentum() {
        std::array<double, 2> momentum = {0.0, 0.0};
        const auto& nodes = mesh2D->getNodes();
        for(const auto& node : nodes) {
            if(!node.isActive) continue;
            
            double rho = std::accumulate(node.distributions.begin(), 
                                       node.distributions.end(), 0.0);
            for(size_t i = 0; i < LatticeTraits<2>::Q; i++) {
                for(size_t d = 0; d < 2; d++) {
                    momentum[d] += node.distributions[i] * 
                                 LatticeTraits<2>::directions[i][d];
                }
            }
        }
        return momentum;
    }

    void addTestIBMNode() {
        std::array<double, 2> pos = {1.5, 1.5};
        std::array<double, 2> vel = {0.1, 0.0};
        solver2D->addIBMNode(pos, vel);
    }
    
    void addTestBoundaries() {
        solver2D->setBoundary(0, BoundaryType::NoSlip);
        solver2D->setBoundary(dims2D[0]-1, BoundaryType::VelocityInlet);
        solver2D->setBoundary(dims2D[0]*dims2D[1]-1, BoundaryType::PressureOutlet);
    }
};

TEST_F(LBMSolverTest, Initialization) {
    solver2D->initialize();
    const auto& nodes = mesh2D->getNodes();
    
    // Check density is 1.0
    for(const auto& node : nodes) {
        if(!node.isActive) continue;
        double rho = std::accumulate(node.distributions.begin(), 
                                   node.distributions.end(), 0.0);
        EXPECT_TRUE(isClose(rho, 1.0));
    }
}

TEST_F(LBMSolverTest, MassConservation) {
    solver2D->initialize();
    double initialMass = computeTotalMass();
    
    solver2D->collideAndStream();
    double finalMass = computeTotalMass();
    
    EXPECT_TRUE(isClose(initialMass, finalMass));
}

TEST_F(LBMSolverTest, MomentumConservation) {
    solver2D->initialize(1.0, {0.1, 0.0});  // Initialize with x-velocity
    auto initialMomentum = computeTotalMomentum();
    
    solver2D->collideAndStream();
    auto finalMomentum = computeTotalMomentum();
    
    EXPECT_TRUE(isClose(initialMomentum[0], finalMomentum[0]));
    EXPECT_TRUE(isClose(initialMomentum[1], finalMomentum[1]));
}

TEST_F(LBMSolverTest, CollisionStep) {
    solver2D->initialize();
    double initialEnergy = 0.0;
    double finalEnergy = 0.0;
    
    const auto& nodes = mesh2D->getNodes();
    for(const auto& node : nodes) {
        if(!node.isActive) continue;
        for(const auto& f : node.distributions) {
            initialEnergy += f * f;
        }
    }
    
    solver2D->collideAndStream();
    
    for(const auto& node : nodes) {
        if(!node.isActive) continue;
        for(const auto& f : node.distributions) {
            finalEnergy += f * f;
        }
    }
    
    EXPECT_LT(finalEnergy, initialEnergy);  // H-theorem
}

TEST_F(LBMSolverTest, StreamingStep) {
    solver2D->initialize(1.0, {0.1, 0.0});
    const auto& nodes = mesh2D->getNodes();
    
    auto initialDist = nodes[4].distributions;  // Center node
    solver2D->collideAndStream();
    
    // Check distribution propagation
    for(size_t i = 1; i < LatticeTraits<2>::Q; i++) {
        size_t neighborIdx = nodes[4].neighbors[i];
        EXPECT_TRUE(isClose(initialDist[i], 
                           nodes[neighborIdx].distributions[LatticeTraits<2>::Q-i]));
    }
}

TEST_F(LBMSolverTest, IBMNodeAddition) {
    addTestIBMNode();
    const auto& ibmNodes = solver2D->getIBMNodes();
    EXPECT_EQ(ibmNodes.size(), 1);
    EXPECT_TRUE(isClose(ibmNodes[0].velocity[0], 0.1));
}

TEST_F(LBMSolverTest, IBMForceCalculation) {
    addTestIBMNode();
    solver2D->initialize(1.0);
    solver2D->collideAndStream();
    
    const auto& ibmNodes = solver2D->getIBMNodes();
    EXPECT_GT(std::abs(ibmNodes[0].force[0]), 0.0);
}

TEST_F(LBMSolverTest, BoundaryConditions) {
    addTestBoundaries();
    solver2D->initialize();
    solver2D->collideAndStream();
    
    const auto& nodes = mesh2D->getNodes();
    // Check bounce-back
    EXPECT_TRUE(isClose(nodes[0].distributions[1], 
                       nodes[0].distributions[3]));
}

TEST_F(LBMSolverTest, MassConservationWithIBM) {
    addTestIBMNode();
    solver2D->initialize();
    double initialMass = computeTotalMass();
    
    solver2D->collideAndStream();
    double finalMass = computeTotalMass();
    
    EXPECT_TRUE(isClose(initialMass, finalMass, 1e-6));
}

TEST_F(LBMSolverTest, VelocityInterpolation) {
    addTestIBMNode();
    solver2D->initialize(1.0, {0.1, 0.0});
    solver2D->collideAndStream();
    
    const auto& ibmNodes = solver2D->getIBMNodes();
    EXPECT_TRUE(isClose(ibmNodes[0].velocity[0], 0.1));
}

TEST_F(LBMSolverTest, BoundaryVelocityProfile) {
    solver2D->setBoundary(0, BoundaryType::VelocityInlet);
    std::array<double, 2> inletVel = {0.1, 0.0};
    solver2D->setInletVelocity(0, inletVel);
    
    solver2D->initialize();
    solver2D->collideAndStream();
    
    const auto& nodes = mesh2D->getNodes();
    auto u = solver2D->computeVelocity(nodes[0].distributions);
    EXPECT_TRUE(isClose(u[0], inletVel[0]));
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}