# Minimal Vulkan Compute project - Linux only, no graphics

load("//:tools/shader.bzl", "glsl_shader")

# Compile GLSL compute shader to SPIR-V
glsl_shader(
    name = "vector_add_shader",
    src = "shaders/vec_add.comp",
)

glsl_shader(
    name = "vector_sub_shader",
    src = "shaders/vec_sub.comp",
)

# Filegroup to ensure shader is included in runfiles
filegroup(
    name = "shaders",
    srcs = [":vector_add_shader", ":vector_sub_shader"],
)


cc_library(
    name = "vk_compute",
    srcs = [
        "vk_compute.cpp",
        "vk_compute.hpp",
    ],
    deps = ["@vulkan_headers//:vulkan_headers"],
    linkopts = [
        "-l:libvulkan.so.1",
        "-ldl",
        "-lpthread",
    ],
)

cc_binary(
    name = "gpu_compute",
    srcs = [
        "main.cpp",
    ],
    data = [":shaders"],
    deps = [":vk_compute"],
)
