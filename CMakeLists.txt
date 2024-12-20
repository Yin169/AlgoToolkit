cmake_minimum_required(VERSION 3.10)
project(FASTSolver)

# Compiler settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra")

# Include directories
include_directories(
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/src/LinearAlgebra/Factorized
    ${PROJECT_SOURCE_DIR}/src/LinearAlgebra/Preconditioner
    ${PROJECT_SOURCE_DIR}/src/LinearAlgebra/Solver
    ${PROJECT_SOURCE_DIR}/src/LinearAlgebra/Krylov
    ${PROJECT_SOURCE_DIR}/src/Obj
)

# Find required packages
find_package(OpenBLAS REQUIRED)
find_package(benchmark REQUIRED)
find_package(GTest REQUIRED)

# Define common test dependencies
set(TEST_DEPENDENCIES
    GTest::gtest 
    GTest::gmock 
    OpenBLAS::OpenBLAS
)

# Test executables
set(TEST_SOURCES
    matrix_obj_test
    basic_test
    debuglogger
    itersolver_test
    LU_test
    KrylovSubspace_test
    ConjugateGradient_test
    SparseMatrixCSCTest
    GMRES_test
)

# Add test executables
foreach(test_source ${TEST_SOURCES})
    add_executable(${test_source} main/${test_source}.cpp)
    target_link_libraries(${test_source} PRIVATE ${TEST_DEPENDENCIES})
    add_test(NAME ${test_source} COMMAND ${test_source})
endforeach()

# Enable testing
enable_testing()
