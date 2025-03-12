rm -rf build
conan install . --build=missing
cd build
cmake --clean-first
cmake .. -DCMAKE_TOOLCHAIN_FILE=Release/generators/conan_toolchain.cmake -DCMAKE_BUILD_TYPE=Release
cmake --build .

./FastPoissonCase
./JetFlow
# python ../application/PostProcess/ShowVTK.py cylinder_flow_0.vtk -f v