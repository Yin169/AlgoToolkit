conan install . --build=missing
bazel clean
bazel --bazelrc=./conan/conan_bzl.rc build --config=conan-config //Obj:matrix_obj_test
bazel test --test_output=all //Obj:matrix_obj_test

bazel --bazelrc=./conan/conan_bzl.rc build --config=conan-config //LinearAlgebra/Factorized:basic_test
bazel test --test_output=all //LinearAlgebra/Factorized:basic_test