cmake_minimum_required(VERSION 3.10)
project(MyCppProject)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 添加头文件搜索路径
include_directories(src)
include_directories(src/LinearAlgebra)
include_directories(src/Obj)

# 找到所有需要的包
find_package(OpenBLAS REQUIRED)
find_package(benchmark REQUIRED)
find_package(GTest REQUIRED)
find_package(fmt REQUIRED)

# 添加可执行文件：matrix_obj_test
add_executable(matrix_obj_test main/matrix_obj_test.cpp)
target_link_libraries(matrix_obj_test PRIVATE GTest::gtest GTest::gmock OpenBLAS::OpenBLAS)

# 添加可执行文件：basic_test
add_executable(basic_test main/basic_test.cpp)
target_link_libraries(basic_test PRIVATE GTest::gtest GTest::gmock OpenBLAS::OpenBLAS)

add_executable(itersolver_test main/itersolver_test.cpp)
target_link_libraries(itersolver_test PRIVATE GTest::gtest GTest::gmock OpenBLAS::OpenBLAS)

# 添加可执行文件：matrix_benchmark_test
add_executable(matrix_benchmark_test main/matrix_benchmark_test.cpp)
target_link_libraries(matrix_benchmark_test PRIVATE benchmark::benchmark OpenBLAS::OpenBLAS)

# 添加可执行文件：basic_benchmark_test
add_executable(basic_benchmark_test main/basic_benchmark_test.cpp)
target_link_libraries(basic_benchmark_test PRIVATE benchmark::benchmark OpenBLAS::OpenBLAS)

# 添加可执行文件：basic_benchmark_test
add_executable(itersolver_benchmark_test main/itersolver_benchmark_test.cpp)
target_link_libraries(itersolver_benchmark_test PRIVATE benchmark::benchmark OpenBLAS::OpenBLAS)

# 添加测试
enable_testing()
add_test(NAME matrix_obj_test COMMAND matrix_obj_test)
add_test(NAME basic_test COMMAND basic_test)
add_test(NAME itersolver_test COMMAND itersolver_test)

# 添加基准测试
add_test(NAME matrix_benchmark_test COMMAND matrix_benchmark_test)
add_test(NAME basic_benchmark_test COMMAND basic_benchmark_test)
add_test(NAME itersolver_benchmark_test COMMAND itersolver_benchmark_test)