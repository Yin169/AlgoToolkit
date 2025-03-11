rm -rf build
conan install . --build=missing
cd build
cmake --clean-first
cmake .. -DCMAKE_TOOLCHAIN_FILE=Release/generators/conan_toolchain.cmake -DCMAKE_BUILD_TYPE=Release
cmake --build .

./exe1

#ctest