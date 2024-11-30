load("@rules_cc//cc:defs.bzl", "cc_test")

cc_library(
    name = "solver_lib",
    hdrs = ["SolverBase.hpp", "IterSolver.hpp"],
    deps = ["//Obj:matrix_lib", "LinearAlgebra/Factorized:basic_lib", "@gtest//:gtest"],
    visibility = ["//visibility:public"],
)

cc_test(
    name = "solver_test",
    srcs = ["solver_test.cpp"],
    deps = ["solver_lib", "@gtest//:gtest"]
)