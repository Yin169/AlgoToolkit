#!/bin/bash

conan install . --build=missing
cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=Release/generators/conan_toolchain.cmake -DCMAKE_BUILD_TYPE=Release
cmake --build .

# ./test
./matrix_obj_test
./basic_test
./matrix_benchmark_test
./basic_benchmark_test

# rm test
rm matrix_obj_test
rm basic_test
rm matrix_benchmark_test
rm basic_benchmark_test