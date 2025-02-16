#include <gtest/gtest.h>
#include <fstream>
#include <string>
#include "Visual.hpp"
#include "MeshObj.hpp"

class VisualTest : public ::testing::Test {
protected:
    std::array<size_t, 2> dims2D = {20, 20};
    std::array<size_t, 3> dims3D = {20, 20, 20};
    std::unique_ptr<MeshObj<double, 2>> mesh2D;
    std::unique_ptr<MeshObj<double, 3>> mesh3D;
    std::string testFile2D = "test_mesh2d";
    std::string testFile3D = "test_mesh3d";
    
    void SetUp() override {
        mesh2D = std::make_unique<MeshObj<double, 2>>(dims2D, 1.0);
        mesh3D = std::make_unique<MeshObj<double, 3>>(dims3D, 1.0);
    }
    
    // void TearDown() override {
    //     std::remove((testFile2D + ".vtk").c_str());
    //     std::remove((testFile3D + ".vtk").c_str());
    // }
    
    std::vector<std::string> readFileLines(const std::string& filename) {
        std::vector<std::string> lines;
        std::ifstream file(filename);
        std::string line;
        while (std::getline(file, line)) {
            lines.push_back(line);
        }
        return lines;
    }
};


TEST_F(VisualTest, HeaderFormat) {
    Visual<double, 2>::writeVTK(*mesh2D, testFile2D);
    auto lines = readFileLines(testFile2D + ".vtk");
    
    EXPECT_GE(lines.size(), 4);
    EXPECT_EQ(lines[0], "# vtk DataFile Version 2.0");
    EXPECT_EQ(lines[2], "ASCII");
    EXPECT_EQ(lines[3], "DATASET UNSTRUCTURED_GRID");
}

// TEST_F(VisualTest, PointData2D) {
//     Visual<double, 2>::writeVTK(*mesh2D, testFile2D);
//     auto lines = readFileLines(testFile2D + ".vtk");
    
//     auto it = std::find_if(lines.begin(), lines.end(),
//         [](const std::string& line) { return line.find("POINTS") != std::string::npos; });
//     EXPECT_NE(it, lines.end());
//     EXPECT_EQ(std::count_if(lines.begin(), lines.end(),
//         [](const std::string& line) { return line.find("0.0") != std::string::npos; }), 9);
// }

TEST_F(VisualTest, CellData2D) {
    Visual<double, 2>::writeVTK(*mesh2D, testFile2D);
    auto lines = readFileLines(testFile2D + ".vtk");
    
    auto it = std::find_if(lines.begin(), lines.end(),
        [](const std::string& line) { return line.find("CELLS") != std::string::npos; });
    EXPECT_NE(it, lines.end());
}

TEST_F(VisualTest, ScalarData) {
    Visual<double, 2>::writeVTK(*mesh2D, testFile2D);
    auto lines = readFileLines(testFile2D + ".vtk");
    
    EXPECT_NE(std::find_if(lines.begin(), lines.end(),
        [](const std::string& line) { return line.find("SCALARS") != std::string::npos; }), 
        lines.end());
}

TEST_F(VisualTest, FileSize) {
    Visual<double, 2>::writeVTK(*mesh2D, testFile2D);
    std::ifstream file(testFile2D + ".vtk", std::ios::binary | std::ios::ate);
    EXPECT_GT(file.tellg(), 0);
}


TEST_F(VisualTest, CellTypes) {
    Visual<double, 2>::writeVTK(*mesh2D, testFile2D);
    auto lines = readFileLines(testFile2D + ".vtk");
    
    auto it = std::find_if(lines.begin(), lines.end(),
        [](const std::string& line) { return line.find("CELL_TYPES") != std::string::npos; });
    EXPECT_NE(it, lines.end());
    
    size_t triangleCount = std::count_if(lines.begin(), lines.end(),
        [](const std::string& line) { return line == "5"; });
    EXPECT_GT(triangleCount, 0);
}

// TEST_F(VisualTest, DataFields) {
//     Visual<double, 2>::writeVTK(*mesh2D, testFile2D);
//     auto lines = readFileLines(testFile2D + ".vtk");
    
//     EXPECT_NE(std::find_if(lines.begin(), lines.end(),
//         [](const std::string& line) { return line.find("SCALARS Level") != std::string::npos; }), 
//         lines.end());
        
//     EXPECT_NE(std::find_if(lines.begin(), lines.end(),
//         [](const std::string& line) { return line.find("SCALARS Active") != std::string::npos; }), 
//         lines.end());
        
//     EXPECT_NE(std::find_if(lines.begin(), lines.end(),
//         [](const std::string& line) { return line.find("VECTORS Velocity") != std::string::npos; }), 
//         lines.end());
// }

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}