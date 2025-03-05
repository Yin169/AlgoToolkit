#include <gtest/gtest.h>
#include "../src/SparseLinearAlgebra/SparseBasic.cpp"
#include <stack>

class SparseBasicTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a simple lower triangular matrix:
        // 1 0 0
        // 2 1 0
        // 3 4 1
        L = SparseMatrixCSC<double>(3, 3);
        L.values = {1.0, 2.0, 3.0, 1.0, 4.0, 1.0};
        L.row_indices = {0, 1, 2, 1, 2, 2};
        L.col_ptr = {0, 3, 5, 6};

        // Create a directed acyclic graph:
        // 0 -> 1 -> 2
        // |         ^
        // +-------->
        dag = SparseMatrixCSC<double>(3, 3);
        dag.values = {1.0, 1.0, 1.0};
        dag.row_indices = {1, 2, 2};
        dag.col_ptr = {0, 2, 3, 3};
    }

    SparseMatrixCSC<double> L;    // Lower triangular matrix
    SparseMatrixCSC<double> dag;  // DAG for testing DFS/Reach
};

TEST_F(SparseBasicTest, DFSTest) {
    std::vector<bool> visited(3, false);
    std::stack<size_t> reach;
    
    SparseLA::dfs(0, dag, visited, reach);
    
    ASSERT_EQ(reach.size(), 3);
    
    // Check topological order
    EXPECT_EQ(reach.top(), 0);
    reach.pop();
    EXPECT_EQ(reach.top(), 1);
    reach.pop();
    EXPECT_EQ(reach.top(), 2);
}

TEST_F(SparseBasicTest, ReachTest) {
    auto reach = SparseLA::Reach(dag);
    
    ASSERT_EQ(reach.size(), 3);
    
    // Verify topological ordering
    std::vector<size_t> order;
    while (!reach.empty()) {
        order.push_back(reach.top());
        reach.pop();
    }
    
    // Check if order respects edges
    for (size_t i = 0; i < order.size(); ++i) {
        for (size_t j = i + 1; j < order.size(); ++j) {
            // Check no edge from order[j] to order[i]
            for (size_t k = dag.col_ptr[order[j]]; k < dag.col_ptr[order[j]+1]; ++k) {
                EXPECT_NE(dag.row_indices[k], order[i]);
            }
        }
    }
}

TEST_F(SparseBasicTest, LSolveTest) {
    VectorObj<double> b(3);
    b[0] = 1.0;
    b[1] = 2.0;
    b[2] = 3.0;
    
    // Make a copy for verification
    VectorObj<double> expected(b);
    
    SparseLA::LSolve(L, b);
    
    // Verify result using forward substitution
    // Ly = b, where L is lower triangular
    expected[0] = expected[0] / L(0, 0);
    expected[1] = (expected[1] - L(1, 0) * expected[0]) / L(1,1);
    expected[2] = (expected[2] - L(2, 0) * expected[0] - L(2, 1) * expected[1]) / L(2, 2);
    
    for (size_t i = 0; i < 3; ++i) {
		std::cout << b[i] << " " << expected[i] << std::endl;
        EXPECT_NEAR(b[i], expected[i], 1e-6);
    }
}

TEST_F(SparseBasicTest, EmptyMatrixTest) {
    SparseMatrixCSC<double> empty(0, 0);
    auto reach = SparseLA::Reach(empty);
    EXPECT_TRUE(reach.empty());
}

TEST_F(SparseBasicTest, SingleElementTest) {
    SparseMatrixCSC<double> single(1, 1);
    single.values = {1.0};
    single.row_indices = {0};
    single.col_ptr = {0, 1};
    
    VectorObj<double> b(1, 2.0);
    SparseLA::LSolve(single, b);
    EXPECT_NEAR(b[0], 2.0, 1e-10);
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}