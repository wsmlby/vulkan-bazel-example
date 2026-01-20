# Minimal Vulkan Compute project - Linux only, no graphics

load("//:tools/shader.bzl", "glsl_shader")

# Compile GLSL compute shader to SPIR-V
glsl_shader(
    name = "vector_add_shader",
    src = "shaders/vector_add.comp",
)

# Filegroup to ensure shader is included in runfiles
filegroup(
    name = "shaders",
    srcs = [":vector_add_shader"],
)

cc_binary(
    name = "gpu_compute",
    srcs = [
        "main.cpp",
        "vk_compute.hpp",
    ],
    data = [":shaders"],
    linkopts = [
        "-l:libvulkan.so.1",
        "-ldl",
        "-lpthread",
    ],
    deps = ["@vulkan_headers//:vulkan_headers"],
)
