load("@rules_cc//cc:defs.bzl", "cc_test")

cc_library(
    name = "basic_lib",
    srcs = ["basic_test.cpp"],
    hdrs = ["basic.hpp"],
    deps = ["//Obj:matrix_lib", "@gtest//:gtest"],
    includes = ["include", "../Obj"],
    visibility = ["//visibility:public"],
)

cc_test(
    name = "basic_test",
    srcs = ["basic_test.cpp"],
    deps = ["basic_lib", "@gtest//:gtest"]
)