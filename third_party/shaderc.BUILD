# Shaderc - build with CMake via genrule
package(default_visibility = ["//visibility:public"])

filegroup(
    name = "shaderc_srcs",
    srcs = glob(["**"]),
)

# Build glslc using cmake
genrule(
    name = "build_glslc",
    srcs = [
        ":shaderc_srcs",
        "@spirv_headers//:all_files",
        "@spirv_tools//:all_files",
        "@glslang//:all_files",
    ],
    outs = ["glslc"],
    cmd = """
        set -e

        # Save the output path as absolute before changing directories
        WORKSPACE_DIR=$$(pwd)
        OUT_PATH="$$WORKSPACE_DIR/$(@)"

        # Setup directories
        BUILD_DIR=$$(mktemp -d)
        INSTALL_DIR=$$(mktemp -d)

        # Copy shaderc sources
        SHADERC_DIR=$$BUILD_DIR/shaderc
        mkdir -p $$SHADERC_DIR
        cp -rL $$WORKSPACE_DIR/external/shaderc/* $$SHADERC_DIR/

        # Copy dependencies into shaderc/third_party
        mkdir -p $$SHADERC_DIR/third_party
        cp -rL $$WORKSPACE_DIR/external/spirv_headers $$SHADERC_DIR/third_party/spirv-headers
        cp -rL $$WORKSPACE_DIR/external/spirv_tools $$SHADERC_DIR/third_party/spirv-tools
        cp -rL $$WORKSPACE_DIR/external/glslang $$SHADERC_DIR/third_party/glslang

        # Build with cmake
        mkdir -p $$BUILD_DIR/build
        cd $$BUILD_DIR/build
        cmake $$SHADERC_DIR \
            -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_INSTALL_PREFIX=$$INSTALL_DIR \
            -DSHADERC_SKIP_TESTS=ON \
            -DSHADERC_SKIP_EXAMPLES=ON \
            -DSHADERC_SKIP_COPYRIGHT_CHECK=ON \
            -DBUILD_SHARED_LIBS=OFF \
            > /dev/null 2>&1

        # Build glslc_exe which produces the actual binary
        cmake --build . --target glslc_exe -j$$(nproc) > /dev/null 2>&1

        # Copy to output using absolute path
        mkdir -p $$(dirname "$$OUT_PATH")
        cp $$BUILD_DIR/build/glslc/glslc "$$OUT_PATH"

        # Cleanup
        rm -rf $$BUILD_DIR $$INSTALL_DIR
    """,
    message = "Building glslc shader compiler",
    visibility = ["//visibility:public"],
)
