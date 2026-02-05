#include "conv_silu.hpp"
#include <stdexcept>

namespace onnxrt {

void ConvSiluOp::prepare(vkcompute::Context& ctx, const std::vector<Tensor*>& inputs,
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

    // Use the fused conv2d_silu shader
    std::string shaderPath = shaderDir + "/conv2d_silu_shader.spv";

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

void ConvSiluOp::record(vkcompute::Sequence& seq, const std::vector<Tensor*>& inputs,
                        const std::vector<Tensor*>& outputs, const Node& node) {
    pipeline_->setPushConstants(&params_, sizeof(params_));

    // Dispatch with 2D tiling: TILE_OW=4 pixels x TILE_K=4 channels per thread
    const uint32_t TILE_OW = 4;
    const uint32_t TILE_K = 4;
    uint32_t tiledOutW = (params_.outW + TILE_OW - 1) / TILE_OW;
    uint32_t tiledK = (params_.K + TILE_K - 1) / TILE_K;
    uint32_t totalTiles = params_.N * tiledK * params_.outH * tiledOutW;
    uint32_t numGroups = (totalTiles + 255) / 256;
    pipeline_->recordTo(seq.cmdBuffer(), numGroups);
}

REGISTER_OPERATOR("ConvSilu", ConvSiluOp);

} // namespace onnxrt
