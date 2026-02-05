load("//:tools/shader.bzl", "glsl_shader")

package(default_visibility = ["//visibility:public"])

# Compile ONNX operator shaders
glsl_shader(name = "add_shader", src = "shaders/add.comp")
glsl_shader(name = "mul_shader", src = "shaders/mul.comp")
glsl_shader(name = "pow_shader", src = "shaders/pow.comp")
glsl_shader(name = "sigmoid_shader", src = "shaders/sigmoid.comp")
glsl_shader(name = "conv2d_shader", src = "shaders/conv2d.comp")
glsl_shader(name = "maxpool_shader", src = "shaders/maxpool.comp")
glsl_shader(name = "resize_shader", src = "shaders/resize.comp")
glsl_shader(name = "concat_shader", src = "shaders/concat.comp")
glsl_shader(name = "slice_shader", src = "shaders/slice.comp")
glsl_shader(name = "transpose_shader", src = "shaders/transpose.comp")

filegroup(
    name = "onnx_shaders",
    srcs = [
        ":add_shader",
        ":mul_shader",
        ":pow_shader",
        ":sigmoid_shader",
        ":conv2d_shader",
        ":maxpool_shader",
        ":resize_shader",
        ":concat_shader",
        ":slice_shader",
        ":transpose_shader",
    ],
)

# ONNX Executor library
cc_library(
    name = "onnx_executor",
    srcs = [
        "onnx_executor.cpp",
        "model.cpp",
        "tensor.cpp",
        "operators/operator_registry.cpp",
        "operators/register_all.cpp",
        "operators/elementwise.cpp",
        "operators/conv.cpp",
        "operators/pool.cpp",
        "operators/resize.cpp",
        "operators/concat.cpp",
        "operators/slice.cpp",
        "operators/transpose.cpp",
        "operators/reshape.cpp",
    ],
    hdrs = [
        "onnx_executor.hpp",
        "model.hpp",
        "tensor.hpp",
        "operators/operator_base.hpp",
        "operators/operator_registry.hpp",
        "operators/register_all.hpp",
        "operators/elementwise.hpp",
        "operators/conv.hpp",
        "operators/pool.hpp",
        "operators/resize.hpp",
        "operators/concat.hpp",
        "operators/slice.hpp",
        "operators/transpose.hpp",
        "operators/reshape.hpp",
    ],
    data = [":onnx_shaders"],
    deps = [
        "//:vk_compute",
        "@onnx//:onnx_cc_proto",
    ],
)

# Example inference binary
cc_binary(
    name = "run_onnx",
    srcs = ["run_onnx.cpp"],
    data = [":onnx_shaders"],
    deps = [":onnx_executor"],
)
