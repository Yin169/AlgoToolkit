cmake_minimum_required(VERSION 3.10)
project(FASTSolver)

# Compiler settings
# set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra")
# set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")
# set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")

# Include directories
include_directories(
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/src/LinearAlgebra/Factorized
    ${PROJECT_SOURCE_DIR}/src/LinearAlgebra/Preconditioner
    ${PROJECT_SOURCE_DIR}/src/LinearAlgebra/Solver
    ${PROJECT_SOURCE_DIR}/src/LinearAlgebra/Krylov
    ${PROJECT_SOURCE_DIR}/src/ODE
    ${PROJECT_SOURCE_DIR}/src/Intergal
    ${PROJECT_SOURCE_DIR}/src/Obj
    ${PROJECT_SOURCE_DIR}/application
    ${PROJECT_SOURCE_DIR}/application/Mesh
    ${PROJECT_SOURCE_DIR}/application/PostProcess
    ${PROJECT_SOURCE_DIR}/application/LatticeBoltz
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
    RungeKutta_test
    GaussianQuad_test
    MeshObj_test
    Visual_test
)

set(EXM_SOURCES
    JetFlow_2D
)

# # Add test executables
# foreach(test_source ${TEST_SOURCES})
#     add_executable(${test_source} test/${test_source}.cpp)
#     target_link_libraries(${test_source} PRIVATE ${TEST_DEPENDENCIES})
#     add_test(NAME ${test_source} COMMAND ${test_source})
# endforeach()

foreach(exm_source ${EXM_SOURCES})
    add_executable(${exm_source} examples/${exm_source}.cpp)
    target_link_libraries(${exm_source} PRIVATE ${TEST_DEPENDENCIES})
    add_test(NAME ${exm_source} COMMAND ${exm_source})
endforeach()

# Enable testing
enable_testing()