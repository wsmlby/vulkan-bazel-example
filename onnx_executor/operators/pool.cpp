#include "pool.hpp"
#include <stdexcept>

namespace onnxrt {

std::vector<Shape> MaxPoolOp::inferShapes(const std::vector<Shape>& inputShapes, const Node& node) const {
    const auto& input = inputShapes[0];
    auto kernelShape = node.getAttr<std::vector<int64_t>>("kernel_shape", {2, 2});
    auto strides = node.getAttr<std::vector<int64_t>>("strides", {1, 1});
    auto pads = node.getAttr<std::vector<int64_t>>("pads", {0, 0, 0, 0});

    int64_t outH = (input[2] + pads[0] + pads[2] - kernelShape[0]) / strides[0] + 1;
    int64_t outW = (input[3] + pads[1] + pads[3] - kernelShape[1]) / strides[1] + 1;

    return {{input[0], input[1], outH, outW}};
}

void MaxPoolOp::prepare(vkcompute::Context& ctx, const std::vector<Tensor*>& inputs,
                        const std::vector<Tensor*>& outputs, const Node& node, const std::string& shaderDir) {
    const auto& input = inputs[0]->shape();
    const auto& output = outputs[0]->shape();
    auto kernelShape = node.getAttr<std::vector<int64_t>>("kernel_shape", {2, 2});
    auto strides = node.getAttr<std::vector<int64_t>>("strides", {1, 1});
    auto pads = node.getAttr<std::vector<int64_t>>("pads", {0, 0, 0, 0});

    params_.N = input[0]; params_.C = input[1]; params_.H = input[2]; params_.W = input[3];
    params_.outH = output[2]; params_.outW = output[3];
    params_.kH = kernelShape[0]; params_.kW = kernelShape[1];
    params_.strideH = strides[0]; params_.strideW = strides[1];
    params_.padH = pads[0]; params_.padW = pads[1];

    pipeline_ = std::make_unique<vkcompute::ComputePipeline>(
        ctx, shaderDir + "/maxpool_shader.spv", 2, 256, sizeof(MaxPoolParams));
    pipeline_->bindBuffers({&inputs[0]->buffer(), &outputs[0]->buffer()});
}

void MaxPoolOp::record(vkcompute::Sequence& seq, const std::vector<Tensor*>& inputs,
                       const std::vector<Tensor*>& outputs, const Node& node) {
    pipeline_->setPushConstants(&params_, sizeof(params_));
    uint32_t total = params_.N * params_.C * params_.outH * params_.outW;
    pipeline_->recordTo(seq.cmdBuffer(), (total + 255) / 256);
}

REGISTER_OPERATOR("MaxPool", MaxPoolOp);

} // namespace onnxrt
