#include <gtest/gtest.h>
#include "SU2MeshAdapter.hpp"
#include "SpectralElementMethod.hpp"

// Test fixture for SU2MeshAdapter
class SU2MeshAdapterTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a simple SU2 mesh file for testing
        std::ofstream mesh_file("test_mesh.su2");
        mesh_file << "NDIME= 2\n";
        mesh_file << "NPOIN= 4\n";
        mesh_file << "0.0 0.0\n";
        mesh_file << "1.0 0.0\n";
        mesh_file << "1.0 1.0\n";
        mesh_file << "0.0 1.0\n";
        mesh_file << "NELEM= 1\n";
        mesh_file << "9 4 0 1 2 3\n"; // Quadrilateral element with 4 nodes
        mesh_file << "NMARK= 1\n";
        mesh_file << "MARKER_TAG= bottom\n";
        mesh_file << "MARKER_ELEMS= 2\n";
        mesh_file << "3 1 0\n"; // Line element (boundary) with 2 nodes
        mesh_file << "3 1 1\n"; // Line element (boundary) with 2 nodes
        mesh_file.close();
    }

    void TearDown() override {
        // Clean up the test mesh file
        std::remove("test_mesh.su2");
    }
};

// Test case for reading the SU2 mesh file
TEST_F(SU2MeshAdapterTest, ReadMeshTest) {
    // Initialize the SpectralElementMethod class
    size_t num_elements = 1;
    size_t polynomial_order = 1;
    size_t num_dimensions = 2;
    std::vector<double> domain_bounds = {0.0, 1.0, 0.0, 1.0};
    std::function<double(const std::vector<double>&)> pde_operator = [](const std::vector<double>&) { return 0.0; };
    std::function<double(const std::vector<double>&)> boundary_condition = [](const std::vector<double>&) { return 0.0; };
    std::function<double(const std::vector<double>&)> source_term = [](const std::vector<double>&) { return 0.0; };

    SpectralElementMethod<double> sem(num_elements, polynomial_order, num_dimensions, domain_bounds, "test_mesh.su2", pde_operator, boundary_condition, source_term);

    // Create the SU2MeshAdapter and read the mesh
    SU2MeshAdapter adapter("test_mesh.su2");
    adapter.readMesh(sem);

    // Verify the number of nodes and elements
    EXPECT_EQ(sem.element_connectivity.size(), 1); // 1 element
    EXPECT_EQ(sem.element_connectivity[0].global_indices.size(), 4); // 4 nodes per element

    // Verify the node coordinates
    std::vector<std::vector<double>> expected_nodes = {
        {0.0, 0.0},
        {1.0, 0.0},
        {1.0, 1.0},
        {0.0, 1.0}
    };
    for (size_t i = 0; i < expected_nodes.size(); ++i) {
        std::vector<double> node_coords = sem.getNodeCoordinates(i);
        EXPECT_NEAR(node_coords[0], expected_nodes[i][0], 1e-10);
        EXPECT_NEAR(node_coords[1], expected_nodes[i][1], 1e-10);
    }

    // Verify the boundary conditions
    // In this test, we assume the boundary condition is applied correctly
    // (the actual boundary condition application logic is simplified in the adapter)
    EXPECT_TRUE(sem.isOnBoundary({0.0, 0.0})); // Node 0 is on the boundary
    EXPECT_TRUE(sem.isOnBoundary({1.0, 0.0})); // Node 1 is on the boundary
}

// Main function to run the tests
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}