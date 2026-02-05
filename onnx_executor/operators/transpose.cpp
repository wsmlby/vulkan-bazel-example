#include "transpose.hpp"

namespace onnxrt {

struct TransposeParams {
    uint32_t totalElements;
    uint32_t ndims;
    uint32_t inputStrides[8];
    uint32_t outputStrides[8];
    uint32_t perm[8];
};

std::vector<Shape> TransposeOp::inferShapes(const std::vector<Shape>& inputShapes, const Node& node) const {
    if (inputShapes.empty()) return {{}};
    const auto& input = inputShapes[0];
    auto perm = node.getAttr<std::vector<int64_t>>("perm", {});

    // Default perm is reverse order
    if (perm.empty()) {
        for (int64_t i = input.size() - 1; i >= 0; i--) {
            perm.push_back(i);
        }
    }

    Shape output(input.size());
    for (size_t i = 0; i < perm.size(); i++) {
        output[i] = input[perm[i]];
    }
    return {output};
}

void TransposeOp::prepare(vkcompute::Context& ctx, const std::vector<Tensor*>& inputs,
                          const std::vector<Tensor*>& outputs, const Node& node, const std::string& shaderDir) {
    perm_ = node.getAttr<std::vector<int64_t>>("perm", {});
    if (perm_.empty()) {
        for (int64_t i = inputs[0]->shape().size() - 1; i >= 0; i--) {
            perm_.push_back(i);
        }
    }
    
    inputShape_ = inputs[0]->shape();
    outputShape_ = outputs[0]->shape();
    totalElements_ = outputs[0]->elementCount();

    pipeline_ = std::make_unique<vkcompute::ComputePipeline>(
        ctx, shaderDir + "/transpose_shader.spv", 2, 256, sizeof(TransposeParams));
    pipeline_->bindBuffers({&inputs[0]->buffer(), &outputs[0]->buffer()});
}

void TransposeOp::record(vkcompute::Sequence& seq, const std::vector<Tensor*>& inputs,
                         const std::vector<Tensor*>& outputs, const Node& node) {
    TransposeParams params{};
    params.totalElements = totalElements_;
    params.ndims = inputShape_.size();
    
    // Compute input strides
    uint32_t stride = 1;
    for (int i = inputShape_.size() - 1; i >= 0; i--) {
        params.inputStrides[i] = stride;
        stride *= inputShape_[i];
    }
    
    // Compute output strides
    stride = 1;
    for (int i = outputShape_.size() - 1; i >= 0; i--) {
        params.outputStrides[i] = stride;
        stride *= outputShape_[i];
    }
    
    // Set permutation
    for (size_t i = 0; i < perm_.size() && i < 8; i++) {
        params.perm[i] = perm_[i];
    }
    
    pipeline_->setPushConstants(&params, sizeof(params));
    pipeline_->recordTo(seq.cmdBuffer(), (totalElements_ + 255) / 256);
}

REGISTER_OPERATOR("Transpose", TransposeOp);

} // namespace onnxrt
