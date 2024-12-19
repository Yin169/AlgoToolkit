cmake_minimum_required(VERSION 3.10)
project(FASTSolver)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(OpenBLAS_DIR "build/Release/generators")

# 添加头文件搜索路径
include_directories(src)
include_directories(src/LinearAlgebra/Factorized)
include_directories(src/LinearAlgebra/Preconditioner)
include_directories(src/LinearAlgebra/Solver)
include_directories(src/LinearAlgebra/Krylov)
include_directories(src/Obj)

# 找到所有需要的包
find_package(OpenBLAS REQUIRED)
find_package(benchmark REQUIRED)
find_package(GTest REQUIRED)
# find_package(pybind11 REQUIRED)

# 添加可执行文件：matrix_obj_test
add_executable(matrix_obj_test main/matrix_obj_test.cpp)
target_link_libraries(matrix_obj_test PRIVATE GTest::gtest GTest::gmock OpenBLAS::OpenBLAS)
# 添加可执行文件：basic_test
add_executable(basic_test main/basic_test.cpp)
target_link_libraries(basic_test PRIVATE GTest::gtest GTest::gmock OpenBLAS::OpenBLAS)

add_executable(debuglogger main/debuglogger.cpp)
target_link_libraries(debuglogger PRIVATE GTest::gtest GTest::gmock OpenBLAS::OpenBLAS)
#
add_executable(itersolver_test main/itersolver_test.cpp)
target_link_libraries(itersolver_test PRIVATE GTest::gtest GTest::gmock OpenBLAS::OpenBLAS)
#
add_executable(LU_test main/LU_test.cpp)
target_link_libraries(LU_test PRIVATE GTest::gtest GTest::gmock OpenBLAS::OpenBLAS)
#
add_executable(KrylovSubspace_test main/KrylovSubspace_test.cpp)
target_link_libraries(KrylovSubspace_test PRIVATE GTest::gtest GTest::gmock OpenBLAS::OpenBLAS)
#
add_executable(ConjugateGradient_test main/ConjugateGradient_test.cpp)
target_link_libraries(ConjugateGradient_test PRIVATE GTest::gtest GTest::gmock OpenBLAS::OpenBLAS)
#
add_executable(SparseMatrixCSCTest main/SparseMatrixCSCTest.cpp)
target_link_libraries(SparseMatrixCSCTest PRIVATE GTest::gtest GTest::gmock OpenBLAS::OpenBLAS)
#
add_executable(MultiGrid_test main/MultiGrid_test.cpp)
target_link_libraries(MultiGrid_test PRIVATE GTest::gtest GTest::gmock OpenBLAS::OpenBLAS)

# add_library(FASTSolver MODULE python/pybind.cpp)
# target_link_libraries(FASTSolver PRIVATE pybind11::module)

# 添加测试
enable_testing()
add_test(NAME matrix_obj_test COMMAND matrix_obj_test)
add_test(NAME SparseMatrixCSCTest COMMAND SparseMatrixCSCTest)
add_test(NAME basic_test COMMAND basic_test)
add_test(NAME debuglogger COMMAND debuglogger)
add_test(NAME itersolver_test COMMAND itersolver_test)
add_test(NAME LU_test COMMAND LU_test)
add_test(NAME KrylovSubspace_test COMMAND KrylovSubspace_test)
add_test(NAME ConjugateGradient_test COMMAND ConjugateGradient_test)
add_test(NAME MultiGrid_test COMMAND MultiGrid_test)
