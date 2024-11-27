conan install . --build=missing
bazel clean
bazel --bazelrc=./conan/conan_bzl.rc build --config=conan-config //Obj:matrix_obj_test
bazel test //Obj:matrix_obj_test