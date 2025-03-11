rm -rf build
conan install . --build=missing
cd build
cmake --clean-first
cmake .. -DCMAKE_TOOLCHAIN_FILE=Release/generators/conan_toolchain.cmake -DCMAKE_BUILD_TYPE=Release
cmake --build .

./PoissonCase
python ../application/PostProcess/ShowVTK.py poisson_solution_sor.vtk -f solution 