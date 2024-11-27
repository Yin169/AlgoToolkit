conan install . --build=missing
bazel clean
bazel --bazelrc=./conan/conan_bzl.rc build --config=conan-config //Obj:matrix_obj_test
bazel test --test_output=all --test_verbose_timeout_warnings //Obj:matrix_obj_test
bazel test --test_output=all --test_verbose_timeout_warnings //Obj:matrix_benchmark_test

bazel --bazelrc=./conan/conan_bzl.rc build --config=conan-config //LinearAlgebra/Factorized:basic_test
bazel test --test_output=all //LinearAlgebra/Factorized:basic_test