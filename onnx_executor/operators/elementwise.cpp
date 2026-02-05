#include "elementwise.hpp"
#include <stdexcept>

namespace onnxrt {

// Helper: Compute broadcast shape
static Shape broadcastShapes(const Shape& a, const Shape& b) {
    Shape result;
    size_t maxDims = std::max(a.size(), b.size());
    result.resize(maxDims);

    for (size_t i = 0; i < maxDims; i++) {
        int64_t dimA = (i < a.size()) ? a[a.size() - 1 - i] : 1;
        int64_t dimB = (i < b.size()) ? b[b.size() - 1 - i] : 1;

        if (dimA == dimB || dimA == 1 || dimB == 1) {
            result[maxDims - 1 - i] = std::max(dimA, dimB);
        } else {
            throw std::runtime_error("Incompatible shapes for broadcasting");
        }
    }
    return result;
}

// ============================================================================
// AddOp
// ============================================================================

std::vector<Shape> AddOp::inferShapes(const std::vector<Shape>& inputShapes, const Node& node) const {
    if (inputShapes.size() < 2) {
        throw std::runtime_error("Add requires 2 inputs");
    }
    return {broadcastShapes(inputShapes[0], inputShapes[1])};
}

void AddOp::prepare(vkcompute::Context& ctx, const std::vector<Tensor*>& inputs,
                    const std::vector<Tensor*>& outputs, const Node& node,
                    const std::string& shaderDir) {
    totalElements_ = static_cast<uint32_t>(outputs[0]->elementCount());
    aElements_ = static_cast<uint32_t>(inputs[0]->elementCount());
    bElements_ = static_cast<uint32_t>(inputs[1]->elementCount());

    std::string shaderPath = shaderDir + "/add_shader.spv";
    pipeline_ = std::make_unique<vkcompute::ComputePipeline>(
        ctx, shaderPath, 3, 256, sizeof(BinaryOpParams));

    pipeline_->bindBuffers({&inputs[0]->buffer(), &inputs[1]->buffer(), &outputs[0]->buffer()});
}

void AddOp::record(vkcompute::Sequence& seq, const std::vector<Tensor*>& inputs,
                   const std::vector<Tensor*>& outputs, const Node& node) {
    BinaryOpParams params{totalElements_, aElements_, bElements_, 0};
    pipeline_->setPushConstants(&params, sizeof(params));

    uint32_t numGroups = (totalElements_ + 255) / 256;
    pipeline_->recordTo(seq.cmdBuffer(), numGroups);
}

REGISTER_OPERATOR("Add", AddOp);

// ============================================================================
// MulOp
// ============================================================================

std::vector<Shape> MulOp::inferShapes(const std::vector<Shape>& inputShapes, const Node& node) const {
    if (inputShapes.size() < 2) {
        throw std::runtime_error("Mul requires 2 inputs");
    }
    return {broadcastShapes(inputShapes[0], inputShapes[1])};
}

void MulOp::prepare(vkcompute::Context& ctx, const std::vector<Tensor*>& inputs,
                    const std::vector<Tensor*>& outputs, const Node& node,
                    const std::string& shaderDir) {
    totalElements_ = static_cast<uint32_t>(outputs[0]->elementCount());
    aElements_ = static_cast<uint32_t>(inputs[0]->elementCount());
    bElements_ = static_cast<uint32_t>(inputs[1]->elementCount());

    std::string shaderPath = shaderDir + "/mul_shader.spv";
    pipeline_ = std::make_unique<vkcompute::ComputePipeline>(
        ctx, shaderPath, 3, 256, sizeof(BinaryOpParams));

    pipeline_->bindBuffers({&inputs[0]->buffer(), &inputs[1]->buffer(), &outputs[0]->buffer()});
}

void MulOp::record(vkcompute::Sequence& seq, const std::vector<Tensor*>& inputs,
                   const std::vector<Tensor*>& outputs, const Node& node) {
    BinaryOpParams params{totalElements_, aElements_, bElements_, 0};
    pipeline_->setPushConstants(&params, sizeof(params));

    uint32_t numGroups = (totalElements_ + 255) / 256;
    pipeline_->recordTo(seq.cmdBuffer(), numGroups);
}

REGISTER_OPERATOR("Mul", MulOp);

// ============================================================================
// PowOp
// ============================================================================

std::vector<Shape> PowOp::inferShapes(const std::vector<Shape>& inputShapes, const Node& node) const {
    if (inputShapes.size() < 2) {
        throw std::runtime_error("Pow requires 2 inputs");
    }
    return {broadcastShapes(inputShapes[0], inputShapes[1])};
}

void PowOp::prepare(vkcompute::Context& ctx, const std::vector<Tensor*>& inputs,
                    const std::vector<Tensor*>& outputs, const Node& node,
                    const std::string& shaderDir) {
    totalElements_ = static_cast<uint32_t>(outputs[0]->elementCount());
    aElements_ = static_cast<uint32_t>(inputs[0]->elementCount());
    bElements_ = static_cast<uint32_t>(inputs[1]->elementCount());

    std::string shaderPath = shaderDir + "/pow_shader.spv";
    pipeline_ = std::make_unique<vkcompute::ComputePipeline>(
        ctx, shaderPath, 3, 256, sizeof(BinaryOpParams));

    pipeline_->bindBuffers({&inputs[0]->buffer(), &inputs[1]->buffer(), &outputs[0]->buffer()});
}

void PowOp::record(vkcompute::Sequence& seq, const std::vector<Tensor*>& inputs,
                   const std::vector<Tensor*>& outputs, const Node& node) {
    BinaryOpParams params{totalElements_, aElements_, bElements_, 0};
    pipeline_->setPushConstants(&params, sizeof(params));

    uint32_t numGroups = (totalElements_ + 255) / 256;
    pipeline_->recordTo(seq.cmdBuffer(), numGroups);
}

REGISTER_OPERATOR("Pow", PowOp);

// ============================================================================
// SigmoidOp
// ============================================================================

std::vector<Shape> SigmoidOp::inferShapes(const std::vector<Shape>& inputShapes, const Node& node) const {
    if (inputShapes.empty()) {
        throw std::runtime_error("Sigmoid requires 1 input");
    }
    return {inputShapes[0]};  // Same shape as input
}

void SigmoidOp::prepare(vkcompute::Context& ctx, const std::vector<Tensor*>& inputs,
                        const std::vector<Tensor*>& outputs, const Node& node,
                        const std::string& shaderDir) {
    totalElements_ = static_cast<uint32_t>(outputs[0]->elementCount());

    std::string shaderPath = shaderDir + "/sigmoid_shader.spv";
    pipeline_ = std::make_unique<vkcompute::ComputePipeline>(
        ctx, shaderPath, 2, 256, sizeof(UnaryOpParams));

    pipeline_->bindBuffers({&inputs[0]->buffer(), &outputs[0]->buffer()});
}

void SigmoidOp::record(vkcompute::Sequence& seq, const std::vector<Tensor*>& inputs,
                       const std::vector<Tensor*>& outputs, const Node& node) {
    UnaryOpParams params{totalElements_, 0.0f, {0, 0}};
    pipeline_->setPushConstants(&params, sizeof(params));

    uint32_t numGroups = (totalElements_ + 255) / 256;
    pipeline_->recordTo(seq.cmdBuffer(), numGroups);
}

REGISTER_OPERATOR("Sigmoid", SigmoidOp);

} // namespace onnxrt
