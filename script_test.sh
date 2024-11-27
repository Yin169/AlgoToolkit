conan install . --build=missing
bazel clean
bazel --bazelrc=./conan/conan_bzl.rc build --config=conan-config //main:demo