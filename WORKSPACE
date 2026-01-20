workspace(name = "gpucompute")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# Vulkan Headers v1.4.339
http_archive(
    name = "vulkan_headers",
    sha256 = "77bc1824f41eea5f3a789ab2927f0916ffa701ed874978c4d1682fd94817e7fb",
    strip_prefix = "Vulkan-Headers-1.4.339",
    urls = ["https://github.com/KhronosGroup/Vulkan-Headers/archive/refs/tags/v1.4.339.tar.gz"],
    build_file_content = """
cc_library(
    name = "vulkan_headers",
    hdrs = glob(["include/vulkan/**/*.h", "include/vulkan/**/*.hpp", "include/vk_video/**/*.h"]),
    includes = ["include"],
    visibility = ["//visibility:public"],
)
""",
)

# SPIRV-Headers (pinned to shaderc v2025.4 dependency)
http_archive(
    name = "spirv_headers",
    sha256 = "494bd30dc13ba798af70edd989f8df82c90757c8ce9433598480f5e00e04c454",
    strip_prefix = "SPIRV-Headers-01e0577914a75a2569c846778c2f93aa8e6feddd",
    urls = ["https://github.com/KhronosGroup/SPIRV-Headers/archive/01e0577914a75a2569c846778c2f93aa8e6feddd.tar.gz"],
    build_file_content = """
filegroup(name = "all_files", srcs = glob(["**"]), visibility = ["//visibility:public"])
""",
)

# SPIRV-Tools (pinned to shaderc v2025.4 dependency)
http_archive(
    name = "spirv_tools",
    sha256 = "7ab19030df38a50cc93be26ebd3913c3e7ef3a118e1ded65a829e63b0c7ccf12",
    strip_prefix = "SPIRV-Tools-19042c8921f35f7bec56b9e5c96c5f5691588ca8",
    urls = ["https://github.com/KhronosGroup/SPIRV-Tools/archive/19042c8921f35f7bec56b9e5c96c5f5691588ca8.tar.gz"],
    build_file_content = """
filegroup(name = "all_files", srcs = glob(["**"]), visibility = ["//visibility:public"])
""",
)

# glslang (pinned to shaderc v2025.4 dependency)
http_archive(
    name = "glslang",
    sha256 = "9537e5896d10bc493775675153687e078d83aead3a5a3b90e9ac28af2bc966fc",
    strip_prefix = "glslang-d213562e35573012b6348b2d584457c3704ac09b",
    urls = ["https://github.com/KhronosGroup/glslang/archive/d213562e35573012b6348b2d584457c3704ac09b.tar.gz"],
    build_file_content = """
filegroup(name = "all_files", srcs = glob(["**"]), visibility = ["//visibility:public"])
""",
)

# Shaderc v2025.4 (includes glslc)
http_archive(
    name = "shaderc",
    sha256 = "8a89fb6612ace8954470aae004623374a8fc8b7a34a4277bee5527173b064faf",
    strip_prefix = "shaderc-2025.4",
    urls = ["https://github.com/google/shaderc/archive/refs/tags/v2025.4.tar.gz"],
    build_file = "@//:third_party/shaderc.BUILD",
)
