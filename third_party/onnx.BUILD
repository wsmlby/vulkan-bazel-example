# Build rules for ONNX protobuf files
# Uses system protoc and links against system libprotobuf

package(default_visibility = ["//visibility:public"])

filegroup(
    name = "onnx_srcs",
    srcs = glob(["**"]),
)

# Compile ONNX proto using system protoc
genrule(
    name = "compile_onnx_proto",
    srcs = [":onnx_srcs"],
    outs = [
        "onnx/onnx.pb.h",
        "onnx/onnx.pb.cc",
    ],
    cmd = """
        set -e
        WORKSPACE_DIR=$$(pwd)
        OUT_DIR=$$(dirname $(location onnx/onnx.pb.h))/..

        # Find the proto file
        PROTO_FILE=$$WORKSPACE_DIR/external/onnx/onnx/onnx.proto

        # Compile with system protoc
        protoc --proto_path=$$WORKSPACE_DIR/external/onnx \
               --cpp_out=$$OUT_DIR \
               $$PROTO_FILE
    """,
    message = "Compiling ONNX protobuf files",
)

# System protobuf library
cc_library(
    name = "protobuf",
    linkopts = ["-lprotobuf"],
)

# ONNX proto library
cc_library(
    name = "onnx_cc_proto",
    srcs = ["onnx/onnx.pb.cc"],
    hdrs = ["onnx/onnx.pb.h"],
    includes = ["."],
    deps = [":protobuf"],
    visibility = ["//visibility:public"],
)
