#include "conv.hpp"
#include <stdexcept>

namespace onnxrt {

std::vector<Shape> ConvOp::inferShapes(const std::vector<Shape>& inputShapes, const Node& node) const {
    if (inputShapes.size() < 2) {
        throw std::runtime_error("Conv requires at least 2 inputs (input and weight)");
    }

    const auto& inputShape = inputShapes[0];   // [N, C, H, W]
    const auto& weightShape = inputShapes[1];  // [K, C/groups, R, S]

    if (inputShape.size() != 4 || weightShape.size() != 4) {
        throw std::runtime_error("Conv requires 4D input and weight tensors");
    }

    int64_t N = inputShape[0];
    int64_t H = inputShape[2];
    int64_t W = inputShape[3];
    int64_t K = weightShape[0];
    int64_t R = weightShape[2];
    int64_t S = weightShape[3];

    // Get attributes with defaults
    auto pads = node.getAttr<std::vector<int64_t>>("pads", {0, 0, 0, 0});
    auto strides = node.getAttr<std::vector<int64_t>>("strides", {1, 1});
    auto dilations = node.getAttr<std::vector<int64_t>>("dilations", {1, 1});

    int64_t padH = pads.size() >= 2 ? pads[0] : 0;
    int64_t padW = pads.size() >= 2 ? pads[1] : 0;
    int64_t strideH = strides.size() >= 1 ? strides[0] : 1;
    int64_t strideW = strides.size() >= 2 ? strides[1] : 1;
    int64_t dilationH = dilations.size() >= 1 ? dilations[0] : 1;
    int64_t dilationW = dilations.size() >= 2 ? dilations[1] : 1;

    // Calculate output dimensions
    int64_t outH = (H + 2 * padH - dilationH * (R - 1) - 1) / strideH + 1;
    int64_t outW = (W + 2 * padW - dilationW * (S - 1) - 1) / strideW + 1;

    return {{N, K, outH, outW}};
}

void ConvOp::prepare(vkcompute::Context& ctx, const std::vector<Tensor*>& inputs,
                     const std::vector<Tensor*>& outputs, const Node& node,
                     const std::string& shaderDir) {
    const auto& inputShape = inputs[0]->shape();   // [N, C, H, W]
    const auto& weightShape = inputs[1]->shape();  // [K, C/groups, R, S]
    const auto& outputShape = outputs[0]->shape(); // [N, K, outH, outW]

    // Get attributes
    auto pads = node.getAttr<std::vector<int64_t>>("pads", {0, 0, 0, 0});
    auto strides = node.getAttr<std::vector<int64_t>>("strides", {1, 1});
    auto dilations = node.getAttr<std::vector<int64_t>>("dilations", {1, 1});
    auto groups = node.getAttr<int64_t>("group", 1);

    params_.N = static_cast<uint32_t>(inputShape[0]);
    params_.C = static_cast<uint32_t>(inputShape[1]);
    params_.H = static_cast<uint32_t>(inputShape[2]);
    params_.W = static_cast<uint32_t>(inputShape[3]);
    params_.K = static_cast<uint32_t>(weightShape[0]);
    params_.R = static_cast<uint32_t>(weightShape[2]);
    params_.S = static_cast<uint32_t>(weightShape[3]);
    params_.outH = static_cast<uint32_t>(outputShape[2]);
    params_.outW = static_cast<uint32_t>(outputShape[3]);
    params_.padH = pads.size() >= 2 ? static_cast<uint32_t>(pads[0]) : 0;
    params_.padW = pads.size() >= 2 ? static_cast<uint32_t>(pads[1]) : 0;
    params_.strideH = strides.size() >= 1 ? static_cast<uint32_t>(strides[0]) : 1;
    params_.strideW = strides.size() >= 2 ? static_cast<uint32_t>(strides[1]) : 1;
    params_.dilationH = dilations.size() >= 1 ? static_cast<uint32_t>(dilations[0]) : 1;
    params_.dilationW = dilations.size() >= 2 ? static_cast<uint32_t>(dilations[1]) : 1;
    params_.groups = static_cast<uint32_t>(groups);
    params_.hasBias = (inputs.size() > 2 && inputs[2] != nullptr) ? 1 : 0;

    std::string shaderPath = shaderDir + "/conv2d_shader.spv";

    // Number of bindings: input, weight, output, bias (4 total)
    int numBindings = params_.hasBias ? 4 : 3;
    pipeline_ = std::make_unique<vkcompute::ComputePipeline>(
        ctx, shaderPath, numBindings, 256, sizeof(Conv2DParams));

    // Bind buffers: input, weight, output, [bias]
    std::vector<vkcompute::Buffer*> buffers;
    buffers.push_back(&inputs[0]->buffer());  // input
    buffers.push_back(&inputs[1]->buffer());  // weight
    buffers.push_back(&outputs[0]->buffer()); // output
    if (params_.hasBias) {
        buffers.push_back(&inputs[2]->buffer());  // bias
    }
    pipeline_->bindBuffers(buffers);
}

void ConvOp::record(vkcompute::Sequence& seq, const std::vector<Tensor*>& inputs,
                    const std::vector<Tensor*>& outputs, const Node& node) {
    pipeline_->setPushConstants(&params_, sizeof(params_));

    // Dispatch one thread per output element
    uint32_t totalOutputs = params_.N * params_.K * params_.outH * params_.outW;
    uint32_t numGroups = (totalOutputs + 255) / 256;
    pipeline_->recordTo(seq.cmdBuffer(), numGroups);
}

REGISTER_OPERATOR("Conv", ConvOp);

} // namespace onnxrt
