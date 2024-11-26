load("@rules_cc//cc:defs.bzl", "cc_test")

cc_library(
    name = "matrix_lib",
    hdrs = ["MatrixObj.hpp"],
    includes = ["include"],
    visibility = ["//visibility:public"],
)

cc_test(
    name = "matrix_obj_test",
    srcs = ["matrix_obj_test.cpp"],
    deps = [":matrix_lib", "@gtest//:gtest"]
)