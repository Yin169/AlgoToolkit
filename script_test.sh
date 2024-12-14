#!/bin/bash

conan install . --build=missing
cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=Release/generators/conan_toolchain.cmake -DCMAKE_BUILD_TYPE=Release
cmake --build .

./demo
./matrix_obj_test
./basic_test
./matrix_benchmark_test
./basic_benchmark_test