# Rule for compiling GLSL compute shaders to SPIR-V

def _glsl_shader_impl(ctx):
    output = ctx.actions.declare_file(ctx.attr.name + ".spv")
    glslc = ctx.file._glslc

    ctx.actions.run(
        inputs = [ctx.file.src, glslc],
        outputs = [output],
        executable = glslc,
        arguments = [
            "-fshader-stage=compute",
            ctx.file.src.path,
            "-o", output.path,
        ],
        mnemonic = "GLSLC",
        progress_message = "Compiling shader %s" % ctx.file.src.short_path,
    )

    return [DefaultInfo(files = depset([output]))]

glsl_shader = rule(
    implementation = _glsl_shader_impl,
    attrs = {
        "src": attr.label(
            allow_single_file = [".comp", ".glsl"],
            mandatory = True,
        ),
        "_glslc": attr.label(
            default = "@shaderc//:build_glslc",
            allow_single_file = True,
        ),
    },
)

def glsl_shaders(name, srcs):
    """Compile multiple GLSL shaders to SPIR-V."""
    all_outputs = []
    for src in srcs:
        shader_name = src.replace("/", "_").replace(".comp", "").replace(".glsl", "")
        glsl_shader(
            name = shader_name,
            src = src,
        )
        all_outputs.append(":" + shader_name)

    native.filegroup(
        name = name,
        srcs = all_outputs,
    )
