#include <gtest/gtest.h>
#include "MeshObj.hpp"
#include "VectorObj.hpp"

class MeshObjTest : public ::testing::Test {
public:
    std::array<size_t, 2> dims2D = {4, 4};
    std::array<size_t, 3> dims3D = {4, 4, 4};
    std::unique_ptr<MeshObj<double, 2>> mesh2D;
    std::unique_ptr<MeshObj<double, 3>> mesh3D;
    
    void SetUp() override {
        mesh2D = std::make_unique<MeshObj<double, 2>>(dims2D, 1.0);
        mesh3D = std::make_unique<MeshObj<double, 3>>(dims3D, 1.0);
    }
    
    bool validateNodePosition(const std::array<double, 2>& pos, double x, double y) {
        return std::abs(pos[0] - x) < 1e-10 && std::abs(pos[1] - y) < 1e-10;
    }
    
    bool validateNodePosition(const std::array<double, 3>& pos, double x, double y, double z) {
        return std::abs(pos[0] - x) < 1e-10 && 
               std::abs(pos[1] - y) < 1e-10 && 
               std::abs(pos[2] - z) < 1e-10;
    }
};

// Grid Initialization Tests
TEST_F(MeshObjTest, Initialize2D) {
    EXPECT_EQ(mesh2D->getNodes().size(), 16);
    auto pos = mesh2D->getNodes()[0].position;
    EXPECT_TRUE(validateNodePosition(pos, 0.0, 0.0));
}

TEST_F(MeshObjTest, Initialize3D) {
    EXPECT_EQ(mesh3D->getNodes().size(), 64);
    auto pos = mesh3D->getNodes()[0].position;
    EXPECT_TRUE(validateNodePosition(pos, 0.0, 0.0, 0.0));
}

// Neighbor Connectivity Tests
TEST_F(MeshObjTest, NeighborConnectivity2D) {
    auto neighbors = mesh2D->getNodes()[5].neighbors;
    EXPECT_EQ(neighbors.size(), 9);  // D2Q9 model
}

TEST_F(MeshObjTest, NeighborConnectivity3D) {
    auto neighbors = mesh3D->getNodes()[21].neighbors;
    EXPECT_EQ(neighbors.size(), 19);  // D3Q19 model
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}