#include "resize.hpp"
#include <cmath>

namespace onnxrt {

std::vector<Shape> ResizeOp::inferShapes(const std::vector<Shape>& inputShapes, const Node& node) const {
    // Output shape determined by scales or sizes input
    return {{}};  // Will be set by executor based on scales/sizes tensor
}

void ResizeOp::prepare(vkcompute::Context& ctx, const std::vector<Tensor*>& inputs,
                       const std::vector<Tensor*>& outputs, const Node& node, const std::string& shaderDir) {
    const auto& input = inputs[0]->shape();
    const auto& output = outputs[0]->shape();

    params_.N = input[0]; params_.C = input[1];
    params_.inH = input[2]; params_.inW = input[3];
    params_.outH = output[2]; params_.outW = output[3];
    params_.scaleH = float(params_.inH) / float(params_.outH);
    params_.scaleW = float(params_.inW) / float(params_.outW);

    pipeline_ = std::make_unique<vkcompute::ComputePipeline>(
        ctx, shaderDir + "/resize_shader.spv", 2, 256, sizeof(ResizeParams));
    pipeline_->bindBuffers({&inputs[0]->buffer(), &outputs[0]->buffer()});
}

void ResizeOp::record(vkcompute::Sequence& seq, const std::vector<Tensor*>& inputs,
                      const std::vector<Tensor*>& outputs, const Node& node) {
    pipeline_->setPushConstants(&params_, sizeof(params_));
    uint32_t total = params_.N * params_.C * params_.outH * params_.outW;
    pipeline_->recordTo(seq.cmdBuffer(), (total + 255) / 256);
}

REGISTER_OPERATOR("Resize", ResizeOp);

} // namespace onnxrt
